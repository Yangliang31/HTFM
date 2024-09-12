# self supervised multimodal multi-task learning network
import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from BertTextEncoder import BertTextEncoder

class HTFM(nn.Module):
    def __init__(self, args):
        super(HTFM, self).__init__()
        # text subnets
        self.aligned = args.need_data_aligned
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_finetune)

        # audio-vision subnets
        audio_in, video_in = args.feature_dims[1:]
        self.audio_model = AuViSubNet(audio_in, args.a_lstm_hidden_size, args.audio_out, \
                            num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        self.video_model = AuViSubNet(video_in, args.v_lstm_hidden_size, args.video_out, \
                            num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)

        # the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(args.text_out + args.video_out + args.audio_out, args.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(args.post_fusion_dim, args.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(args.post_fusion_dim, 1)

        self.post_text_dropout = nn.Dropout(p=args.post_text_dropout)
        self.post_text_layer_1 = nn.Linear(768, 128)

        self.gru = nn.GRU(256,128)
        self.gru1 = nn.GRU(128,128)
        self.gru2 = nn.GRU(384,384)
        self.multiattn = nn.MultiheadAttention(embed_dim=128, num_heads=4)

        self.batchnorm = nn.BatchNorm1d(2, affine=False)
        self.batchnorm1 = nn.BatchNorm1d(3, affine=False)

        self.fusion_dropout = nn.Dropout(p=args.post_video_dropout)
        self.fusion_layer_1 = nn.Linear(in_features=384, out_features=256, bias = False)
        self.fusion_layer_2 = nn.Linear(in_features=256,out_features=1)
        



    def new_method(self,t,a,v):
        ta,_ = self.multiattn(t,a,a)
        ta,_ = self.gru1(ta)
        tav,_ = self.multiattn(ta,v,v)
        tav,_ = self.gru1(tav)
        return tav        ###############

    def forward(self, text, audio, video):
        audio, audio_lengths = audio
        video, video_lengths = video

        mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze().int().detach().cpu()
        text = self.text_model(text)[:,0,:]

        if self.aligned:
            audio = self.audio_model(audio, text_lengths)
            video = self.video_model(video, text_lengths)
        else:
            audio = self.audio_model(audio, audio_lengths)
            video = self.video_model(video, video_lengths)

        # # text
        text_h = self.post_text_dropout(text)
        text_h = F.relu(self.post_text_layer_1(text_h), inplace=False)


        F1 = self.new_method(text_h,audio,video)
        
        F2 = torch.cat([text_h, audio, video], dim=-1)
        F2 = self.post_fusion_dropout(F2)
        F2 = F.relu(self.post_fusion_layer_1(F2), inplace=False)
        F2,_ = self.gru1(F2)

        text_s,_ = self.multiattn(text_h,text_h,text_h)
        audio_s,_ = self.multiattn(audio,audio,audio)
        video_s,_ = self.multiattn(video,video,video)
        F3 = torch.cat([text_s, audio_s, video_s], dim=-1)
        F3 = self.post_fusion_dropout(F3)
        F3 = F.relu(self.post_fusion_layer_1(F3), inplace=False)
        F3,_ = self.gru1(F3)

        Final = torch.cat([F1, F2,F3], dim=-1)
        #Final,_ = self.gru2(Final)
        Final = self.fusion_layer_1(Final)
        Final = self.fusion_layer_2(Final)

        res = {
            'M': Final, 
            'output1': F1,
        }
        return res

class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(AuViSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        packed_sequence = pack_padded_sequence(x, lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1

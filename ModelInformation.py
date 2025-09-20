import os
import glob
import torch
import torch.nn as nn
# import torch.nn.functional as F
import pandas as pd
import gc
import csv
from tqdm import tqdm
import datetime


import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class CustomTransformer(nn.Module):
    def __init__(self, num_geohash, d_model):
        super(CustomTransformer, self).__init__()
        # print("initial d_model= {}.".format(d_model))
        self.embedding = nn.Embedding(num_embeddings=num_geohash, embedding_dim=d_model)
        self.transformer = nn.Transformer(d_model=d_model, 
                                          num_encoder_layers=2, 
                                          num_decoder_layers=2, 
                                          dropout=0.1,
                                          dim_feedforward=512, 
                                          batch_first=True, 
                                          nhead=4)

        self.positional_encoding = PositionalEncoding(d_model, dropout=0)
        self.MLPModel = torch.nn.Sequential(
            nn.Linear(d_model, 128),
            torch.nn.ReLU(),
            nn.Linear(128, num_geohash)
        )

    def forward(self, src, tgt):
        # src_key_padding_mask = CustomTransformer.get_key_padding_mask(src)
        # tgt_key_padding_mask = CustomTransformer.get_key_padding_mask(tgt)
        # tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[1])
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[1])
        # tgt_mask = torch.triu(torch.ones(tgt.size()[1], tgt.size()[1]), diagonal=1).bool()
        src = self.embedding(src.long())
        tgt = self.embedding(tgt.long())
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        # print(tgt.shape)
        # print(src.dtype, tgt.dtype)
        out_t = self.transformer(src, tgt,
                                 tgt_mask=tgt_mask,
                                #  src_key_padding_mask=src_key_padding_mask,
                                #  tgt_key_padding_mask=tgt_key_padding_mask
                                )
        
        out = self.MLPModel(out_t)
        return out_t, out
    
    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == 2] = -torch.inf
        return key_padding_mask


# # 模型实例化。
# ctModel = CustomTransformer(num_geohash=gNum_geohash, d_model=gParameters.get('g_d_model'))
# loss_function_crossentropy = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(ctModel.parameters(), lr=3e-3)

# # 仅以下为模型测试代码。 
# feature, out = ctModel(x_grid, y_grid)
# print(feature.shape, out.shape)
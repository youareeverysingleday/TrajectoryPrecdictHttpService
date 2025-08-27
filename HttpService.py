import pandas as pd
import numpy as np
import time
import os
import csv

import uvicorn
from pydantic import BaseModel, Field
from typing import List, Any
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, JSONResponse
import asyncio
import torch
import torch.nn as nn
# import requests

# 提供接口。
# uvicorn Service:app --reload
app = FastAPI(docs_url=None)

@app.get('/')
def read_root():
    return {'hello': 'world'}

app.mount('/static', StaticFiles(directory='static'), name='static')
@app.get('/docs', include_in_schema=False)
async def cunstom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title='Predict Trajectory',
        swagger_js_url='/static/swagger-ui-bundle.js',
        swagger_css_url='/static/swagger-ui.css',
        
        swagger_favicon_url="/static/favicon-16x16.png",
        )

@app.get("/favicon.png", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.png")


class Input(BaseModel):
    clientID: int # 使用者的身份ID。
    userID: int # 被预测对象的ID。
    topK: int # 需要提供几个备选位置。
    length: int # 区域ID和时间戳的长度。
    regionID : List[int]
    timestamp : List[str]

class Output(BaseModel):
    nextTimeLocation : int #预测的下一个 Δt 的位置。 Δt default is 20min 。

def WriteLog(clientID, userID, nextLocation, length, historyTrajectories, log_path):
    """_summary_
    写日志
    Args:
        clientID (_type_): 查询者的ID。
        userID (_type_): 被预测对象的ID。
        nextLocation (_type_): 对于被预测对象的预测的下一个时刻的位置。
    """
    # 长度的阈值。
    lengthThreshold = 100
    # 组成需要记录的信息。
    log_infor = [clientID, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), userID, nextLocation, length]
    # 历史轨迹信息保存下来。
    # 如果长度小于 lengthThreshold，那么后面的填0 。
    # 如果长度大于 lengthThreshold，那么在 lengthThreshold 处截断。
    # 如果长度等于 lengthThreshold，那么就正好保存。
    if length < lengthThreshold :
        # 链表中的数据越到后面越新靠近当前时间。
        historyTrajectories = [0] * (lengthThreshold - len(historyTrajectories)) + historyTrajectories
    elif length > lengthThreshold:
        historyTrajectories = historyTrajectories[0:lengthThreshold]
    else:
        pass
    for element in historyTrajectories:
            log_infor.append(element)
    
    # 如果文件存在则追加写入。如果文件不存在则新建一个文件。
    if os.path.isfile(log_path):
        with open(log_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(log_infor)
    else:
        with open(log_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            columnsName = ['clientID', 'time', 'userID', 'nextLocation', 'length']
            for i in range(lengthThreshold):
                columnsName += ['loc{}'.format(i)]
            writer.writerow(columnsName)
            writer.writerow(log_infor)

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
    def __init__(self, num_geohash, d_model, nhead=4, num_layers=2, ff_dim=512, dropout=0.1, padding_idx=2):
        super(CustomTransformer, self).__init__()
        self.padding_idx = padding_idx
        # print("initial d_model= {}.".format(d_model))
        self.embedding = nn.Embedding(num_embeddings=num_geohash, embedding_dim=d_model)
        self.dropout_emb = nn.Dropout(p=dropout)
        self.transformer = nn.Transformer(d_model=d_model, 
                                          num_encoder_layers=num_layers, 
                                          num_decoder_layers=num_layers, 
                                          dropout=dropout,
                                          dim_feedforward=ff_dim, 
                                          batch_first=True, 
                                          nhead=nhead)

        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.MLPModel = torch.nn.Sequential(
            nn.Linear(d_model, 128),
            torch.nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_geohash)
        )

    def get_key_padding_mask(self, tokens):
        """
        返回 True 表示该位置为 padding（被 mask）
        输入：tokens: [batch, seq_len]
        输出：mask: [batch, seq_len]，bool
        """
        return (tokens == self.padding_idx)
    
    def forward(self, src, tgt):
        # src_key_padding_mask = CustomTransformer.get_key_padding_mask(src)
        # tgt_key_padding_mask = CustomTransformer.get_key_padding_mask(tgt)
        src_key_padding_mask = self.get_key_padding_mask(src).to(device)
        tgt_key_padding_mask = self.get_key_padding_mask(tgt).to(device)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[1]).to(device)
        # tgt_mask = torch.triu(torch.ones(tgt.size()[1], tgt.size()[1]), diagonal=1).bool()
        # src = self.embedding(src.long())
        # tgt = self.embedding(tgt.long())
        src = self.dropout_emb(self.embedding(src.long()))
        tgt = self.dropout_emb(self.embedding(tgt.long()))
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        # print(tgt.shape)
        # print(src.dtype, tgt.dtype)
        out_t = self.transformer(src, tgt,
                                 tgt_mask=tgt_mask,
                                 src_key_padding_mask=src_key_padding_mask,
                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                 memory_key_padding_mask=src_key_padding_mask  # 注意：memory 是 encoder 的输出
                                )
        
        out = self.MLPModel(out_t)
        return out_t, out

gNum_geohash = 21000
g_d_model = 28

async def ExecuteModel(input: Input):
    print(input.userID)
    await asyncio.sleep(10) 
    # 设置日志。
    time_day_stamp = time.strftime('%Y%m%d', time.localtime())
    log_save_path = f'./log/' + 'log_{}.csv'.format(time_day_stamp)

    # 加载模型并且预测。
    # ----------------start-----------------
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # 获得历史轨迹。
    HistoryGrid = torch.tensor(input.regionID)
    # 获取topk的值。
    topK= input.topK

    # 加载模型。
    predictModel = CustomTransformer(num_geohash=gNum_geohash, d_model=g_d_model)
    checkpoint = torch.load('./model.pth')
    predictModel.load_state_dict(checkpoint['model_state_dict'])

    src = HistoryGrid.unsqueeze(0).to(device)
    tgt = src[:, -1].unsqueeze(0).to(device)

    _, out = predictModel(src, tgt)
    _, output_topk_index = torch.topk(out[0, -1, :], k=topK, dim=-1)

    nextLocation = 925
    # ----------------end-----------------

    WriteLog(clientID=input.clientID, userID=input.userID, nextLocation=nextLocation, length=input.length, 
             historyTrajectories=input.regionID, log_path=log_save_path)

    output = Output(nextTimeLocation=nextLocation)
    return output

@app.post("/predict")
async def predict(input:Input):
    # result = await Test(input.identity)
    result = await ExecuteModel(input)
    return result
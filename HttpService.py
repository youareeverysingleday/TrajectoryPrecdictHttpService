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
import requests

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


async def ExecuteModel(input: Input):
    print(input.userID)
    await asyncio.sleep(10) 
    time_day_stamp = time.strftime('%Y%m%d', time.localtime())
    log_save_path = f'./log/' + 'log_{}.csv'.format(time_day_stamp)

    nextLocation = 925

    WriteLog(clientID=input.clientID, userID=input.userID, nextLocation=nextLocation, length=input.length, 
             historyTrajectories=input.regionID, log_path=log_save_path)

    output = Output(nextTimeLocation=nextLocation)
    return output

@app.post("/predict")
async def predict(input:Input):
    # result = await Test(input.identity)
    result = await ExecuteModel(input)
    return result
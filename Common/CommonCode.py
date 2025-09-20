# from ast import keyword

# 时间使用库。
import datetime
import time
from typing import Any
import math

# 数据处理使用库。
import numpy as np
import pandas as pd

# tensorflow的数据处理。
# import tensorflow as tf

# 稀疏矩阵使用库。
# from scipy import sparse

# 绘图使用库。
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d

# 文件处理使用库。
import os
import imageio.v2 as imageio

# numpy.array 转化为 csv。
import csv

# 读取全局变量。
import json
# 解决json无法解析注释的问题。
import re

# 绘制轨迹
import folium

# 用于实现对三维数据进行归一化。
from sklearn.preprocessing import MinMaxScaler
# 划分数据集。
# import sklearn.model_selection as sk_model_selection
# 数据集归一化。
# import sklearn.preprocessing as sk_preprocessing

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
import glob

import transbigdata as tbd


class CommonTimer:
    """记录多次运行时间"""
    def __init__(self) -> None:
        self.times = []
        self.start()
        
    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

class SparseMatrix:
    """_summary_
    还没有完成。可能会使用tensorflow来完成。因为涉及到大量计算。
    生成稀疏矩阵。
    获取稀疏矩阵中的值。
    """
    def __init__(self, data, rows, columns) -> None:
        # self.Matrix = sparse.csr_matrix((data, (rows, columns)),  
        #                           shape=(len(rows), len(columns)))
        pass
        
    def __call__(self, matrix) -> Any:
        self.Matrix = matrix

    def GetItem(self, row_index, column_index):
        # Get row values
        row_start = self.Matrix .indptr[row_index]
        row_end = self.Matrix .indptr[row_index + 1]
        row_values = self.Matrix .data[row_start:row_end]

        # Get column indices of occupied values
        index_start = self.Matrix .indptr[row_index]
        index_end = self.Matrix .indptr[row_index + 1]

        # contains indices of occupied cells at a specific row
        row_indices = list(self.Matrix .indices[index_start:index_end])

        # Find a positional index for a specific column index
        value_index = row_indices.index(column_index)

        if value_index >= 0:
            return row_values[value_index]
        else:
            # non-zero value is not found
            return 0

# 创建对运算时间进行计算的函数。
def DisplayStartInfo(description=""):
    """
    :description: 显示开始时间。
    :param None {type: None}: 
    :return startTime {type: datetime.datetime}{count: 1}: 返回当前时间。
    """
    print("-------------------------Start---{}----------------------".format(description))
    startTime = datetime.datetime.now()
    print(startTime.strftime('%Y-%m-%d %H:%M:%S'))
    return startTime


def DisplayCompletedInfo(description="", startTime=datetime.datetime.now(), 
                         isDisplayTimeConsumed=False):
    """
    :description: 显示结束时间信息。
    :param startTime {type: datetime.datetime} {default: datetime.datetime.now()}:: 显示开始时间。default值搭配着isDisplayTimeConsumed=False一起使用，这个时候可以不输出消耗时间。
    :param isDisplayTimeConsumed {type: bool} {default: False}: 是否显示消耗的时间，默认值不显示。
    :return None {count: 0}: 
    """
    if isDisplayTimeConsumed==True:
        print('Time consumed:', str(datetime.datetime.now() - startTime).split('.')[0])
    print("Completed at " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ".\n")
    print("-------------------------Completed---{}----------------------".format(description))

def DisplaySeparator(description=""):
    print("---{}------------------------------------------------------\n".format(description))


def List2Csv(listData, CsvPath, axis=0):
    """_summary_
    将一个List保存为一个csv格式的文件。
    Args:
        listData (list): 需要保存的List数据。
        CsvPath (string): 保存为csv的路径（包含文件名）。
        axis (int, optional): 0表示按行保存list；1表示按列保存list。
            特别是对于输入的是单行的list的时候，如果希望保存的形状是(1,:)，那么需要axis=0；
            如果希望保存的形状是(:,1)，那么需要axis=1；保存的时候需要注意方向。Defaults to 0.
    """
    # print(listData)
    if axis == 1:
        if len(listData) > 0:
            # columns=columnsName, 
            df = pd.DataFrame(listData)
            # mode='w', header=1, index=0
            df.to_csv(CsvPath, mode='w', encoding='utf-8', header=0, index=0)
    else:
        if len(listData) > 0:
            # columns=columnsName, 
            df = pd.DataFrame([listData])
            # mode='w', header=1, index=0
            df.to_csv(CsvPath, mode='w', encoding='utf-8', header=0, index=0)

# 测试数据集。
# columnsName = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
# TestData = pd.DataFrame(np.random.randint(0,100,size=(20, 10)), columns=columnsName)
# TestData

def GetMultiplyDatasMaximumValue(Dataset, MaximumValueAmount, axis=0):
    """_summary_
    返回的是输入Dataset具体的最大的值。
    Args:
        Dataset (DataFrame): 需求计算最大值的数据集。
        MaximumValueAmount (int): 每行或者每列需要获取最大值的数量。需要小于Dataset的对应的形状。
        axis (int, optional): 需要进行计算的方向。0表示按行计算，1表示按列计算. Defaults to 0.

    Returns:
        MaximunsData or None (DataFrame or None): 返回记录最大值的DataFrame。
    """
    
    # 按行取最大值。
    if axis == 0:
        # 因为是按行取，所以与列值进行比较。
        if MaximumValueAmount > Dataset.shape[1]:
            print("CommonCode.py GetMultiplyDatasMaximumValue \
                  Error. MaximumValueAmount {} is bigger than Dataset.shape[0] {}.".
                  format(MaximumValueAmount, Dataset.shape[0]))
            return None

        # 行号循环变量。
        i = 0
        print(Dataset.shape[1])
        MaximunsData = pd.DataFrame(np.zeros((Dataset.shape[0], MaximumValueAmount)), index=Dataset.index)
        for index, row in Dataset.iterrows():
            MaximunsData.iloc[i, :] = pd.DataFrame(row).nlargest(MaximumValueAmount, index, keep='first').T
            i += 1

        return MaximunsData
    elif axis == 1:
        # 因为是按列取，所以与行值进行比较。
        if MaximumValueAmount > Dataset.shape[0]:
            print("CommonCode.py GetMultiplyDatasMaximumValue \
                  Error. MaximumValueAmount {} is bigger than Dataset.shape[1] {}.".
                  format(MaximumValueAmount, Dataset.shape[1]))
            return None

        # 列号循环变量。
        i = 0
        # 按列取最大值。
        MaximunsData = pd.DataFrame(np.zeros((MaximumValueAmount, Dataset.shape[1])), columns=Dataset.columns)
        for columnName, column in Dataset.items():
            # print(column.shape)
            TempColumn = pd.DataFrame(column).nlargest(MaximumValueAmount, columnName, keep='first')
            # print(TempColumn)
            # 注意这里一定需要删除index，因为如果在原始TempColumn中包含index，
            # 那么赋值给MaximunsData时会按照TempColumn和MaximunsData的index相同的进行赋值。直接导致MaximunsData的结果不正确。
            # 不论使用哪种对列进行赋值的方法，这一步reset_index都是必须的。
            TempColumn.reset_index(drop=True, inplace=True)
            # print(TempColumn.shape)
            
            MaximunsData[MaximunsData.columns[i]] = TempColumn
            # print(MaximunsData)
            i += 1
        return MaximunsData
    else:
        print("Axis is error.")
        return None


def GetMultiplyDataMaximumIndexorColumnName(Dataset, MaximumValueAmount, axis=0):
    """_summary_
    返回的是输入的Dataset的最大值的index或者column name。
    Args:
        Dataset (DataFrame): 需求计算最大值的数据集。
        MaximumValueAmount (int): 每行或者每列需要获取最大值的数量。需要小于Dataset的对应的形状。
        axis (int, optional): 需要进行计算的方向。0表示按行计算，1表示按列计算. Defaults to 0.

    Returns:
        MaximunsData or None (DataFrame or None): 返回记录最大值的DataFrame。
    """
    
    # 按行取最大值。
    if axis == 0:
        # 因为是按行取，所以与列值进行比较。
        if MaximumValueAmount > Dataset.shape[1]:
            print("CommonCode.py GetMultiplyDataMaximumIndexorColumnName \
                  Error. MaximumValueAmount {} is bigger than Dataset.shape[0] {}.".
                  format(MaximumValueAmount, Dataset.shape[0]))
            return None

        # 行号循环变量。
        i = 0
        # print(Dataset.shape[1])
        MaximunsFlag = pd.DataFrame(np.zeros((Dataset.shape[0], MaximumValueAmount)), index=Dataset.index)
        # print(Dataset.index)
        for index, row in Dataset.iterrows():
            MaximunsFlag.iloc[i, :] = pd.DataFrame(row).nlargest(MaximumValueAmount, index, keep='first').T.columns
            i += 1

        return MaximunsFlag
    elif axis == 1:
        # 因为是按列取，所以与行值进行比较。
        if MaximumValueAmount > Dataset.shape[0]:
            print("CommonCode.py GetMultiplyDataMaximumIndexorColumnName \
                  Error. MaximumValueAmount {} is bigger than Dataset.shape[1] {}.".
                  format(MaximumValueAmount, Dataset.shape[1]))
            return None

        # 列号循环变量。
        i = 0
        # 按列取最大值。
        MaximunsFlag = pd.DataFrame(np.zeros((MaximumValueAmount, Dataset.shape[1])), columns=Dataset.columns)
        for columnName, column in Dataset.items():
            # print(column.shape)
            # TempColumn = pd.DataFrame(column).nlargest(MaximumValueAmount, columnName, keep='first')
            # # print(zz)
            # # 注意这里一定需要删除index，因为如果在原始TempColumn中包含index，
            # # 那么赋值给MaximunsData时会按照TempColumn和MaximunsData的index相同的进行赋值。直接导致MaximunsData的结果不正确。
            # TempColumn.reset_index(drop=True, inplace=True)
            # # print(zz.shape)
            
            MaximunsFlag[MaximunsFlag.columns[i]] = pd.DataFrame(column).nlargest(MaximumValueAmount, columnName, keep='first').index.index
            print(MaximunsFlag)
            i += 1
        return MaximunsFlag
    else:
        print("Axis is error.")
        return None
    
def GetNonzeroIndexorColumnName(Dataset, axis=0):
    """_summary_
    现在这个函数主要用于在推荐系统中。从test数据集中找出单一用户访问过的所有item。
    注意返回的不是Dataset中的数据值，而是对应非零值的行号或者列名。
    Args:
        Dataset (pandas.DataFrame): 输入需要进行排查的Dataframe。

    Returns:
        nonZero (pandas.DataFrame): 二维列表。因为Dataset中每行非零值的数目可能不同，所以只能用二维list来存储。
    """

    # 先用二维list进行缓存，最后转化为dataframe返回。
    # 注意初始化的时候是一维的，每次追加的也是一维list，最终是二维list。
    nonZero = []

    DatasetColumnName = Dataset.columns
    DatasetIndexName = Dataset.index

    if axis == 0:
        for index, row in Dataset.iterrows():
            # 这个转换的操作步骤如下：
            # 1. 将row转换为numpy.array；注意此时将所有dataframe中对应的列名丢弃了。
            # 2. 通过numpy.array的nonzero()方法取出该行中所有非0值的索引。
            # 3. 因为2中取出的值是tuple，所以通过[0]来取出所有索引。
            # 4. 将索引对应DataSet中的列名取出来，也是通过DatasetColumnName（这不是一个list类型）来获取的；
            #    此时获取的在recommendation system中也就是item的编号，后面需要计算RS的性能的。
            # 5. 将获取的列名通过list格式进行存储。注意因为二维list才能存储不同长度的数据。
            #    后面在使用nonZero时是通过dataframe的nonan()方法来进行处理的。
            nonZero.append(DatasetColumnName[(np.array(row).nonzero())[0]].tolist())
    elif axis == 1:
        for columnName, column in Dataset.iteritems():
            # 注意按照列进行遍历之后再存储到nonZero中时方向变了。
            nonZero.append(DatasetIndexName[(np.array(column).nonzero())[0]].tolist())
    else:
        print('CommonCode.py GetNonzeroIndexorColumnName() Error. input axis {} is Error.'.format(axis))
        return None
    
    # 后面在使用nonZero时是通过dataframe的nonan()方法来进行处理的。
    # 之所以需要转换为dataframe是因为dataframen可以携带index。在recommendation system中也就对应着用户ID。
    nonZero_df = pd.DataFrame(nonZero, index=Dataset.index.tolist())
    return nonZero_df

def ReadHugeFile(inputPath):
    """_summary_
    按行读取超大文件。
    Args:
        inputPath (_type_): 大文件存在的路径。

    Yields:
        string: 返回大文件中的每一行。
    """
    # 一定需要注意，对于读取报错的行进行忽略。
    with open(inputPath, 'r', encoding='UTF-8', errors='ignore') as file:
        for line in file:
            try:
                yield line
            except:
                pass


class PrivateDebug():
    
    def __init__(self) -> None:
        # 暂时还没有使用Levels。
        self.DisplayLevels = ['All', 'Information', 'Key', 'Debug', 'Warning', 'Error']
        self.Keyword = ""
        
    def AddDisplayLevel(self, Keyword):
        if Keyword not in self.DisplayLevels:
            self.DisplayLevels.append(Keyword)
        return True

    def SetDisplayKeyword(self, keyword):
        self.Keyword = keyword
    
    def OutputContent(self, keyword, msg, *args):
        """_summary_

        Args:
            keyword (string): 区分打印的关键字。
            msg (string): 需要输出的内容。
            *args (tuple): 元组，可以带多个参数。
        """
        if keyword.lower() == "nodisplay":
            return 
        if (self.Keyword == keyword):
            # print(msg)
            # print(args)
            print("{} value is {}".format(msg, args))
        # 通过*args传进来的参数一定是tuple类型。所以显示类型没有意义，显示shape会报错。
        # print("{} type is {}".format(msg, type(args)))
        # print("{} shape is {}".format(msg, args.shape))
    
class GenerateAnimation():
    def __init__(self, x, y, z, x_label, y_label, z_label, 
                 title, figureSavePath, figureMainName, gifSavePath, 
                 figsize=(16, 12), dpi=64, duration=0.2, type='3d',
                 startAngle=30, endAngle=70, interval=4) -> None:
        """_summary_

        Args:
            x (dataframe): x轴数据集。
            y (dataframe): y轴数据集。
            z (dataframe): z轴数据集。
            x_label (string): x轴标签。
            y_label (string): y轴标签。
            z_label (string): z轴标签。
            title (string): 图片名称。
            figureSavePath (string): 生成的每张图片保存路径，这个路径不包含文件名。
            figureMainName (string): 生成的每张图片是需要取一个主体的名称，
                然后和figureSavePath结合一起使用。
            gifSavePath (string): 动画保存路径，注意这个路径包含文件名。
            figsize (tuple, optional): 每张图片的大小. Defaults to (16, 12).
            dpi (int, optional): 每张图片的分辨率. Defaults to 64.
            duration (float, optional): 动画中每张图片的时间间隔（秒）. Defaults to 0.2.
            type (str, optional): 是3D还是2D动画. Defaults to '3d'.
            startAngle (int, optional): 3D动画开始时的视角角度. Defaults to 30.
            endAngle (int, optional): 3D动画结束时的视角角度. Defaults to 70.
            interval (int, optional): 3D动画视角角度变化的间隔. Defaults to 4.
        """
        self.x = x
        self.y = y
        self.z = z
        self.x_label = x_label
        self.y_label = y_label
        self.z_label = z_label
        self.title = title
        self.duration = duration
        self.figureSavePath = figureSavePath
        self.figureMainName = figureMainName
        self.gifSavePath = gifSavePath
        self.figsize = figsize
        self.dpi = dpi
        self.type = type
        self.startAngle = startAngle
        self.endAngle = endAngle
        self.interval = interval
        pass
    
    
    def __call__(self):
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus']=False

        ims = []

        for angle in range(self.startAngle, self.endAngle, self.interval):
            plt.clf()

            # 创建画布。
            # 创建画布必须放在循环里面，不然这个画布只对第1张图片有效。
            fig = plt.figure(figsize=(16, 12), dpi=48)
            # 创建绘图区域。
            ax = plt.axes(projection='3d')

            # 设置绘图区域的相关参数。
            ax.set_title('每年特征统计图')
            ax.set_xlabel('weekofyear')
            ax.set_ylabel('NodeID')
            ax.set_zlabel('statistic')
            ax.axis('auto')

            ax.view_init(30, angle)
            # 绘制散点图。
            im = ax.scatter3D(self.x, self.y, self.z).findobj()

            pictureName = self.figureSavePath + self.figureMainName + str(angle) + '.png'
            plt.savefig(pictureName, dpi=96)
            ims.append(im)
        
        path = self.figureSavePath
        pictureNames = os.listdir(path)
        list_of_im_paths = []
        for pictureName in pictureNames:
            list_of_im_paths.append(self.figureSavePath + pictureName)
        # print(list_of_im_paths)

        ims = [imageio.imread(f) for f in list_of_im_paths]
        imageio.mimwrite(self.gifSavePath, ims, duration = self.duration)
        print("Generate Animation has Completed.")
        
# 将numpy.narray的3维数据保存为csv格式。
def np_3d_to_csv(data, 
                 path, 
                 datatype='float'):
    a2d = data.reshape(data.shape[0], -1)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(a2d)

# 从3维数据中读取numpy.narray格式。
def np_3d_read_csv(path='./Data/Output/StayMatrx/{}.csv',
                   shape=(-1, 128, 3),
                   datatype='float'):
    """_summary_

    Args:
        path (str, optional): 文件存储路径. Defaults to './Data/Output/StayMatrx/{}.csv'.
        shape (tuple, optional): _description_. Defaults to (-1, 128, 3).
        datatype (str, optional): _description_. Defaults to 'float'.

    Returns:
        numpy.array: 返回读取的3维数据。
    """
    # 从csv文件读取2D数组
    with open(path, "r") as f:
        reader = csv.reader(f)
        a2d = np.array(list(reader)).astype(datatype)

    # 将2D数组转换为3D数组
    a = a2d.reshape(shape)
    # print(a.shape)
    return a

def data_split_onedimension(sequence, windows_length=100):
    """_summary_
    将一个一维向量通过滑动窗口划分为样本和标签。
    标签是取样本之后的第一个值。
    Args:
        sequence (numpy.array): 一维数组，也就是向量。
        windows_length (int, optional): 滑动窗口的长度. Defaults to 100.

    Returns:
        np.array(x) (numpy.array): 一维数组，样本。
        np.array(y) (numpy.array): 一维数组，标签。
    """
    x = []
    y = []
    
    for i in range(len(sequence)):
        labelIndex = i + windows_length
        if labelIndex > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:labelIndex], sequence[labelIndex]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)

def data_split_twodimension(sequence, windows_length=100, step_length=1):
    """_summary_
    将一个二维矩阵通过滑动窗口划分为样本和标签。
    标签取样本之后的第一个值。
    这个序列是按行进行的时间序列。
    Args:
        sequence (numpy.array): 三维矩阵，类似于文本的形状。
        windows_length (int, optional): 滑动窗口的长度. Defaults to 100.
        step_length (int, optional): 移动滑动窗口的步长. Defaults to 1.
    Returns:
        np.array(x) (numpy.array): 二维数组，样本。
        np.array(y) (numpy.array): 二维数组，标签。
    """
    x = []
    y = []
    
    for i in range(math.ceil(len(sequence)/step_length)):
        labelIndex = step_length * i + windows_length
        if labelIndex > len(sequence) - 1:
            break
        # sequence[i:labelIndex, :], sequence[labelIndex, :]
        seq_x, seq_y = sequence[step_length*i:labelIndex, :], sequence[labelIndex, :]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)

def data_split_onedimension(sequence, windows_length=100):
    """_summary_
    将一维序列按照滑动窗口转化为矩阵。这里没有使用步长这个参数。
    Args:
        sequence (numpy.narray): 输出的一维序列。实际上就是一个向量。
        windows_length (int, optional): 滑动窗口长度. Defaults to 100.

    Returns:
        x_original (numpy.narray): 原始的一维向量变为矩阵。
        x (numpy.narray): 样本。
        y (numpy.narray): 标签。
    """
    
    firstDimension = sequence.shape[0]
    if firstDimension % windows_length != 0:
        sequence = np.pad(sequence, (0, windows_length -(firstDimension % windows_length)),
                            'constant', constant_values=(0, 0))
    x_original = sequence.reshape(-1, windows_length)

    x = np.array(sequence[:-windows_length]).reshape(-1, windows_length)
    y = np.array(sequence[windows_length:]).reshape(-1, windows_length)
    # gridMatrix.shape
    return x_original, x, y


def data_split_twodimension_to_matrix(sequence, windows_length=100):
    """_summary_
    将一个二维矩阵时序的数据处理为符合transformer输入的情况。
    也就是将这个时序数据处理为2个类似NLP的矩阵。
    源矩阵是按滑动窗口长度获取的。

    单个样本是前100个特征作为输入，后面一个作为输出。
    所有的样本，x比原始的少最后windows_length个元素；y比原始的少前windows_length个元素。
    
    标签取样本之后的第一个值。
    这个序列是按行进行的时间序列。
    Args:
        sequence (numpy.array): 二维矩阵，按时间排序的序列形式。
        windows_length (int, optional): 滑动窗口的长度. Defaults to 100.

    Returns:
        x_original (numpy.array): 原始数据直接升维，同时形状按最后两个维度进行调整。
        x (numpy.array): 三维矩阵，生成样本数据。
        y (numpy.array): 三维矩阵，生成标签数据。
    """
    firstDimension = sequence.shape[0]
    lastDimension = sequence.shape[-1]

    # 因为要变形，所以对于无法除尽的维度进行填充。
    # ((0, windows_length -(firstDimension % windows_length)),(0,0))填充的含义是：
    # 第一个维度（也就是行）之前添加0行，之后添加windows_length -(firstDimension % windows_length) 行。
    # 因为是时序数据，所以只用添加行即可。
    # 第二个维度（也就是列）之前添加0列，之后添加0列。
    # 如果维度能够被滑动窗口的长度整除，那么就不需要填充。
    if firstDimension % windows_length != 0:
        sequence = np.pad(sequence, ((0, windows_length -(firstDimension % windows_length)),(0,0)),
                        'constant', constant_values=(0,0)) 
    
    x_original = sequence.reshape(-1, windows_length, lastDimension)
    
    # 实现的思路实际上是通过滑动窗口来实现的。
    # 但具体实现上就相当于第滑动窗口长度的元素作为y的开始位置，之后的都是y的。
    # x的是从0开始，直到-window的位置终止。
    # 也就是将滑动窗口的运算转化为了矩阵的位置运算。所以下面的代码没有问题。
    # 就输出的形状而言，x比原始的少最后windows_length个元素；y比原始的少前windows_length个元素。
    x = np.array(sequence[:-windows_length, :]).reshape(-1, windows_length, lastDimension)
    y = np.array(sequence[windows_length:, :]).reshape(-1, windows_length, lastDimension)

    return x_original, x, y

def data_twodimension_to_threedimension_series(sequence, delete_index,windows_length=100, step_length=1):
    """_summary_
    将一个二维矩阵时序的数据处理为符合分类问题的输入的情况。
    也就是将这个时序数据处理为输入是多行的矩阵，输出是一个分类值。
    源矩阵是按滑动窗口长度获取的。
    
    标签取样本之后的第一个值。
    这个序列是按行进行的时间序列。
    Args:
        sequence (numpy.array): 三维矩阵，时间序列数据。
        delete_index (int): 在生成数据的时候需要将样本中的grid列删除。
        windows_length (int, optional): 滑动窗口的长度. Defaults to 100.
        step_length (int, optional): 移动滑动窗口的步长. Defaults to 1.
    Returns:
        np.array(x) (numpy.array): 二维矩阵，样本数据。
        np.array(y) (numpy.array): 标签。
    """
    x = []
    y = []
    
    for i in range(math.ceil(len(sequence)/step_length)):
        labelIndex = step_length * i + windows_length
        # 这里之所以需要减1是因为最后一列需要作为标签值取出来。
        # 如果是单纯滑动窗口加步长的平移是不用减1的。
        if labelIndex > len(sequence) - 1 :
            break
        seq_x, seq_y = sequence[step_length*i:labelIndex, :], sequence[labelIndex, :]
        # 删除grid列。
        seq_x = np.delete(seq_x, obj=delete_index, axis=1)
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


def InverseVector(scaler, one_vetcor_of_originalmatrix, vertor):
    """_summary_
    还原一个基于归一化了的原始数据预测特征值的向量。
    Args:
        scaler (object): 一个sklearn.preprocessing中定义的归一化对象。
        one_vetcor_of_originalmatrix (numpy.ndarray): 归一化之后原始数据中的选取的任意一个向量。一行或者一列，不用使用整个原始数据。
        vertor (numpy.ndarray): 需要还原的向量。

    Returns:
        numpy.ndarray: 被还原的向量。
    """
    temp = np.row_stack((one_vetcor_of_originalmatrix, vertor))
    return scaler.inverse_transform(temp)[-1, :]


def ReadJson(JsonPath='./Parameters.json'):
    """_summary_
    解决json文件中包含注释无法解析的问题。
    Args:
        JsonPath (str, optional): _description_. Defaults to './Parameters.json'.
    """
    with open(JsonPath, 'r', encoding='utf-8') as file:
        json_data = file.read()
        pattern = r'//.*?$|/\*.*?\*/'
        json_data = re.sub(pattern=pattern, repl=' ',
                        string=json_data, flags=re.MULTILINE|re.DOTALL)
        
        parsed_data = json.loads(json_data)
        return parsed_data

class JSONConfig:
    """_summary_
    对在parameters.json中存储的全局变量进行操作。
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self._load_json()

    def _load_json(self):
        """加载 JSON 文件"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save(self):
        """保存数据到 JSON 文件"""
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

    def get(self, key, default=None):
        """获取 JSON 变量"""
        return self.data.get(key, default)

    def set(self, key, value):
        """设置 JSON 变量并保存"""
        self.data[key] = value
        self.save()

    def delete(self, key):
        """删除 JSON 变量并保存"""
        if key in self.data:
            del self.data[key]
            self.save()

def visualizationFunction(history, FirstSavePath, SecondSavePath, preTitle=""):
    """_summary_
    对模型的历史数据进行可视化。
    Args:
        history (tensorflow.history): 输入的历史记录。
        preTitle (str, optional): 标题名称. Defaults to "".
    """
    keys = []
    for k in history.history.keys():
        keys.append(k)
    values = []
    for v in history.history.values():
        values.append(v)

    epochs = range(len(values[0]))

    plt.plot(epochs, values[0], 'r', label="Training {}".format(keys[0]))
    plt.plot(epochs, values[1], 'b', label="Validation {}".format(keys[1]))
    plt.title(preTitle + " Training and validation {}".format(keys[1]))
    plt.legend()
    plt.savefig(FirstSavePath)
    plt.figure()

    plt.plot(epochs, values[2], 'r', label="Training {}".format(keys[2]))
    plt.plot(epochs, values[3], 'b', label="Validation {}".format(keys[3]))
    plt.title(preTitle + " Training and validation {}".format(keys[3]))
    plt.legend()
    plt.savefig(SecondSavePath)
    plt.figure()
    plt.show()

def DisplaySingleUserHistoryTrajectory(userID, cols=['lat', 'lon'],
                                       dataPath='../Data/Input/Stay/{}.csv',
                                       savePath='../Pictures/Test/{}Traj.html', zoom_start=12):
    """_summary_
    显示单个用户的历史轨迹。
    现在生成地图的中心点不是使用的固定中心点，而是使用的轨迹经纬度平均值作为中心点。
    北京参考中心点为：[39.1289, 117.3539]。
    Args:
        userID (str): 用户的的ID，用于读取指定用户的轨迹.
        cols (list, optional): _description_. Defaults to ['lat', 'lon'].
        dataPath (str, optional): 轨迹数据存储的位置. Defaults to '../Data/Input/Stay/{}.csv'.
        savePath (str, optional): 生成的估计可视化文件存放位置. Defaults to '../Pictures/Test/{}Traj.html'.
        zoom_start (int, optional): 地图放大的倍率. Defaults to 12.
    """
    data = pd.read_csv(dataPath.format(userID), usecols=cols)
    map = folium.Map(location=[data[cols[0]].mean(), data[cols[1]].mean()], zoom_start=zoom_start, title='{} trajectory'.format(userID))
    trajectory = []
    
    for name, row in data.iterrows():
        trajectory.append([row[cols[0]], row[cols[1]]])

    folium.PolyLine(trajectory, color="blue", weight=2.5, opacity=1).add_to(map)
    map.save(savePath.format(userID))

def ScalerThreeDimensionMatrix(data, feature_range=(-1, 1)):
    """_summary_
    对三维矩阵的最后2个维度进行归一化。
    对于三维数据，先转成2维再归一化再还原。也就是将最后两维展开。
    Args:
        data (numpy.array/ pandas.Dataframe): 输入的数据。
        feature_range (tuple, optional): 归一化时值的范围. Defaults to (-1, 1).

    Returns:
        numpy.array: 输出完成归一化之后的三维矩阵。
    """
    dim0, dim1, dim2 = data.shape
    # dim0, dim2

    reshaped_tensor = data.reshape((dim0, dim1 * dim2))
    return MinMaxScaler(feature_range=feature_range).fit_transform(reshaped_tensor).reshape((dim0, dim1, dim2))

def GetTensorBytes(tensor, name='tensor'):
    """_summary_
    返回tensor占用的内存大小。也可以说明占用的显存大小。
    打印的结果是以Mbyte 为单位。
    tensor.nelement() 是tensor中元素的个数。
    tensor.element_size() 是tensor中单个元素占用的内存大小。
    
    Args:
        tensor (torch.tensor): 输入teorch的tensor。
        name (str, optional): 输入tensor的名称. Defaults to 'tensor'.
    """
    print('{} memory size is {} bytes.'.format(name, tensor.nelement() * tensor.element_size() /1024/1024 ))


def CantorPairingFunction(x, y):
    """_summary_
    先对x,y使用折叠函数，然后再计算2个数的cantor配对函数的值。
    Args:
        x (int): 整数。
        y (int): 整数。

    Returns:
        int: 返回cantor配对数。
    """
    if x >= 0:
        x = 2 * x
    else:
        x = 2 * abs(x) - 1
    
    if y >= 0:
        y = 2 * y
    else:
        y = 2 * abs(y) - 1

    return ((x + y) * (x + y + 1) // 2 + y)

def CantorPairingInverseFunction(z):
    """_summary_
    先计算cantor配对函数反函数，然后x,y使用折叠反函数。
    注意是loncol在前，latcol在后。
    Args:
        z (int): 两个数的cantor配对数值。

    Returns:
        x (int): 整数。loncol。
        y (int): 整数。latcol。
    """
    if z < 0 :
        print('CantorPairingInverseFunction input z is out of range.')
        return 0, 0
    
    w = (math.sqrt(8 * z + 1) - 1) // 2
    t = w * (w + 1) // 2
    y = z - t
    x = w - y
    
    if x % 2 == 0:
        x = x / 2
    else:
        x = -((x + 1) / 2)
    
    if y % 2 == 0:
        y = y / 2
    else:
        y = -((y + 1) / 2)

    return int(x), int(y)

def GenerateGrid(df, lonColName='loncol', latColName='latcol'):
    """_summary_
    将 康托 配对函数应用到dataframe上，生成grid。
    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    df['grid'] = CantorPairingFunction(df[lonColName], df[latColName])
    return df

def RecoverLoncolLatcol(df, gridColName='grid'):
    """_summary_
    将 康托 配对函数的反函数应用到dataframe上，生成行号和列号。
    Args:
        df (_type_): _description_
        gridColName (str, optional): grid在dataframe中的列名. Defaults to 'grid'.

    Returns:
        _type_: _description_
    """
    df['loncol'], df['latcol']= CantorPairingInverseFunction(df[gridColName])
    return df

def Calculate2DConvOutputShape(inputShape, kennel_size, padding, stride, describe=''):
    """_summary_
    计算二维卷积的输出形状。没有计算batch 和 out_channel。
    Args:
        inputShape (_type_): _description_
        kennel_size (_type_): _description_
        padding (_type_): _description_
        stride (_type_): _description_
    
    Returns:
        (h_out, w_out): 输出数据的形状
    """
    
    h_out = math.floor((inputShape[0] - kennel_size[0] + padding[0]) / stride[0]) + 1
    w_out = math.floor((inputShape[1] - kennel_size[1] + padding[1]) / stride[1]) + 1
    print('{} Conv2D Output height is {}, width is {}.'.format(describe, h_out, w_out))
    return (h_out, w_out)

def Calculate2DPoolMaxOutputShape(inputShape, kennel_size, padding, stride, dilation=1, describe=''):
    """_summary_
    计算二维池化的输出形状。没有计算batch 和 out_channel。
    Args:
        inputShape (_type_): _description_
        kennel_size (_type_): _description_
        padding (_type_): _description_
        stride (_type_): _description_
        dilation (int, optional): _description_. Defaults to 1.

    Returns:
        (h_out, w_out): 输出数据的形状
    """
    h_out = math.floor((inputShape[0] + 2 * padding[0] - dilation * (kennel_size[0] - 1) - 1) / stride[0]) + 1
    w_out = math.floor((inputShape[1] + 2 * padding[1] - dilation * (kennel_size[1] - 1) - 1) / stride[1]) + 1
    print('{} PoolMax2D output height is {}, width is {}.'.format(describe, h_out, w_out))
    return (h_out, w_out)

def SwapColumns_TwoDimension(tensor):
    """_summary_
    注意只能对2维矩阵中最后一个维度的两列进行交换。
    为了计算距离，需要将经纬度的位置交换。geopy是经度在前，纬度在后。

    Args:
        tensor (torch.tensor): 需要被交换位置的tensor。

    Returns:
        torch.tensor: 被交换了位置的tensor。
    """
    # 交换一个矩阵中两列的值。
    # .clone()
    tensor = tensor.clone()
    tensor[[0, 1]] = tensor[[1, 0]]
    return tensor

def SwapColumns_ThreeDimension(tensor):
    """_summary_
    注意只能对3维矩阵的最后一个维度中的两列进行交换。
    为了计算距离，需要将经纬度的位置交换。geopy是经度在前，纬度在后。

    Args:
        tensor (torch.tensor): 需要被交换位置的tensor。

    Returns:
        torch.tensor: 被交换了位置的tensor。
    """
    # 交换一个矩阵中两列的值。
    # .clone()
    tensor = tensor.clone()
    tensor[:, :, [0, 1]] = tensor[:, :, [1, 0]]
    return tensor

def haversine_distance(coords1, coords2):
    """
    使用 Haversine 公式批量计算两组经纬度之间的距离（单位：公里）。
    不使用 geopy.distance 来进行计算。因为无法利用torch的加速。
    纬度latitude在前面, 经度longitude在后面。
    
    参数：
    - coords1: [N, 2] 的 tensor，表示 (纬度, 经度)
    - coords2: [N, 2] 的 tensor，表示 (纬度, 经度)
    
    返回：
    - distances: [N] 的 tensor，存储每对坐标之间的距离（单位 km）
    """
    R = 6371.0  # 地球半径（单位：km）

    # 经纬度转换为弧度.
    lat1, lon1 = torch.deg2rad(coords1[:, 0]), torch.deg2rad(coords1[:, 1])
    lat2, lon2 = torch.deg2rad(coords2[:, 0]), torch.deg2rad(coords2[:, 1])
    # print("lat2 shape {}, lat1 shape {}.".format(lat2.shape, lat1.shape))

    # Haversine 公式
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    
    return R * c  # 返回距离（单位：km）

# # 示例数据
# # coords1 = torch.tensor([[39.9042, 116.4074], [34.0522, -118.2437]])
# coords1 = torch.tensor([[39.9042, 116.4074], [34.0522, -118.2437]])  # 北京 & 洛杉矶
# coords2 = torch.tensor([[39.9052, 116.4074], [51.5074, -0.1278]])  # 纽约 & 伦敦

# distances = cc.haversine_distance(coords1, coords2)
# print(distances)  # 输出距离（单位：km）

class MultipleInputDataset(Dataset):
    """_summary_
    自定义有多个输入的模型dataset类，比如transformer模型。
    区别于MLP，transformer模型需要两个输入参数。
    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, src, tgt, label):
        super(MultipleInputDataset, self).__init__()
        self.src = torch.tensor(src)
        self.tgt = torch.tensor(tgt)
        self.label = torch.tensor(label)

    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, index):
        # return super().__getitem__(index)
        # return {
        #     "src":self.src[index],
        #     "tgt":self.tgt[index]
        # }
        return (self.src[index], self.tgt[index], self.label[index])

def GetDataLoader(**kwargs):
    """_summary_
    x_path, x_2_path, y_path, batch_size=16, train_size=0.8, isMultipleInput=False
    将数据加载到loader中便于训练。
    Args:
        x_path (string): 数据存储的路径。
        x_2_path (string): 需要多个输入的模型的第二个数据的路径。
        y_path (string): 标签数据存储的路径。
        batch_size (int, optional): 每个epoch中样本的数量. Defaults to 16.
        train_size (float, optional): 训练集. Defaults to 0.8.
        isMultipleInput (bool, optional): 是否为多输入的模型，比如transformer . Defaults to False.

    Returns:
        _type_: _description_
    """
    x = torch.load(kwargs['x_path'], weights_only=False)
    if not torch.is_tensor(x):
        x = torch.from_numpy(x)

    y = torch.load(kwargs['y_path'], weights_only=False)
    if not torch.is_tensor(y):
        y = torch.from_numpy(y)
    print("file path {} ,x shape{}, y shape {}.".format(kwargs['x_path'], x.shape, y.shape))

    if kwargs['isMultipleInput'] == True:
        x_2 = torch.load(kwargs['x_2_path'], weights_only=False)
        if not torch.is_tensor(x_2):
            x_2 = torch.from_numpy(x_2)
        dataset = MultipleInputDataset(x, x_2, y)
    else:
        dataset = TensorDataset(x, y)

    # 将数据集分为训练集和测试集
    # train_dataset, test_dataset 
    # trainLoader, testLodaer
    train_dataset, test_dataset = random_split(dataset, 
                                           lengths=[int(kwargs['train_size'] * len(dataset)), 
                                                    len(dataset) - int(kwargs['train_size'] * len(dataset))],
                                           generator=torch.Generator().manual_seed(0))
    
    trainLoader = DataLoader(train_dataset, batch_size=kwargs['batch_size'], shuffle=False, num_workers=0)
    testLodaer = DataLoader(test_dataset, batch_size=kwargs['batch_size'], shuffle=False, num_workers=0)
    return trainLoader, testLodaer


def load_checkpoint(start_epoch, model, optimizer, checkpoint_path):
    """_summary_
    载入模型模型和优化器参数。
    从所有的断点训练文件中选择最新的一个。
    Args:
        start_epoch (_type_): 被保存断点训练文件中最后一次训练的epoch值。
        model (_type_): 模型。
        optimizer (_type_): 优化器。
        checkpoint_path (str, optional): 保存断点文件的目录..

    Returns:
        _type_: _description_
    """
    checkpoint_pattern = os.path.join(checkpoint_path, 'checkpoint_*.pth')
    # 获取所有的 checkpoint 文件。
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    # 如果存在检查点，加载并继续训练
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)

        print(f"load latest checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"from {start_epoch} start to train...")
    else:
        print('Cant get any checkpoint file. Restart training.')
        start_epoch = 0

    return start_epoch

def DisplayModelTrainResult(logPath, x_col, y_train_col, y_test_col, 
                            plot_label_train, plot_label_test, 
                            x_label, y_label, title, 
                            savePath, extremeFlag='min', offset=1):
    """_summary_
    显示从模型保存在 loss_offset_log.csv 的数据。
    Args:
        logPath (string): loss_offset_log.csv 文件的保存路径。
        x_col (string): x 轴对应在 loss_offset_log.csv 中的列名称。
        y_train_col (string): 训练数据对应在 loss_offset_log.csv 中的列名称。
        y_test_col (string): 测试数据对应在 loss_offset_log.csv 中的列名称。
        plot_label_train (string): 在 plt.plot() 函数中train数据对应 label 参数填写的值。也就是图线的示例名。
        plot_label_test (string): 在 plt.plot() 函数中test数据对应label 参数填写的值。也就是图线的示例名。
        x_label (string): x轴名称。
        y_label (string): y轴名称。
        title (string): 图片名称。
        savePath (string): 图片保存路径。
        extremeFlag (string): 显示极值点是显示极大值还是极小值。
        offset (int): 显示极值信息时的偏移。
    """
    # 读取 CSV 数据
    loss_offset_log= pd.read_csv(logPath)

    # 显示极小值。
    if extremeFlag == 'min':
        # 测试数据集的损失最小值。
        pdextreme = loss_offset_log.iloc[loss_offset_log[y_test_col].idxmin()]
        extremeepoch = pdextreme[x_col]
        extremeTest = pdextreme[y_test_col]
        # show_text=f'{int(minepoch)}: {minTest:.2f}'
        show_text = f'x:{int(extremeepoch)}\ny:{extremeTest:.2f}'
    else:
        # 显示极大值。
        # 测试数据集的损失最大值。
        pdextreme = loss_offset_log.iloc[loss_offset_log[y_test_col].idxmax()]
        extremeepoch = pdextreme[x_col]
        extremeTest = pdextreme[y_test_col]
        # show_text=f'{int(minepoch)}: {minTest:.2f}'
        show_text = f'x:{int(extremeepoch)}\ny:{extremeTest:.2f}'

    # 可视化 Loss 曲线 Epoch", "TrainLoss", "TrainOffset", "TestLoss", "TestOffset"]
    # marker="o",
    plt.figure(figsize=(8, 5))
    plt.plot(loss_offset_log[x_col], loss_offset_log[y_train_col],  linestyle="-", color="b", label=plot_label_train)
    plt.plot(loss_offset_log[x_col], loss_offset_log[y_test_col],  linestyle="-", color="r", label=plot_label_test)

    plt.scatter(extremeepoch, extremeTest, color='red', s=25) 
    if extremeFlag == 'min':
        plt.annotate(show_text, xytext=(extremeepoch-1, extremeTest+extremeTest/10),xy=(extremeepoch, extremeTest))
    else:
        plt.annotate(show_text, xytext=(extremeepoch-1, extremeTest-extremeTest/10),xy=(extremeepoch, extremeTest))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(savePath)
    plt.show()



def GenerateAllGridMapping(Bounds, 
                           mappingColumnName = 'grid_mapping',
                           mappingSavePath='./Data/Output/all_grid_mapping.csv'):
    """_summary_
    按照地图信息将所有区域grid映射到一个合理范围。
    直接使用康托尔配对函数时将 loncols 和 latcols 直接计算得到的值会极大的超过实际网格的数量。
    所以对得到的值进行了一次缩放（也就是映射）到了网格数量对应的范围中。
    Args:
        Bounds (_type_): transbigdata 所需的区域范围参数。
        mappingColumnName (str, optional): 映射生成列的列名. Defaults to 'grid_mapping'.
        mappingSavePath (str, optional): 映射保存路径. Defaults to './Data/Output/all_grid_mapping.csv'.

    Returns:
        _type_: _description_
    """
    GeoParameters = tbd.area_to_params(Bounds, accuracy = 1000, method='rect')
    n_lon = int((Bounds[2] - Bounds[0]) / GeoParameters['deltalon'])
    n_lat = int((Bounds[3] - Bounds[1]) / GeoParameters['deltalat'])

    # 获取所有栅格编号 [LONCOL, LATCOL]
    loncols = list(range(n_lon))
    latcols = list(range(n_lat))
    # 生成所有loncol , latcol。
    all_grid_df = pd.DataFrame([[lon, lat] for lon in loncols for lat in latcols], columns=['loncol', 'latcol'])
    # 生成grid。
    all_grid_df = all_grid_df.apply(GenerateGrid , lonColName='loncol', latColName='latcol', axis=1)

    GridColumnData = pd.DataFrame(all_grid_df.loc[:, 'grid'])
    # 生成列名。这一步不能删除，没有添加列名，就是一个series，后面就无法重新排序了。
    GridColumnData.columns = ['grid']
    # 去重之后共计9396个区域，数量没有变化。
    Grid_duplicated = GridColumnData.drop_duplicates()
    # 重新排序。查看前10个样本，感觉实际也没有做任何操作。可能已经排过序了。
    Grid_duplicated = Grid_duplicated.sort_values(by='grid', ascending=True)
    # 重置Index。
    Grid_duplicated = Grid_duplicated.reset_index(drop=True)
    # 将index 定义为新的映射列名。
    Grid_duplicated[mappingColumnName] = Grid_duplicated.index
    # 将映射全部加1，因为需要将0作为异常值赋值给未知区域对应的映射。
    # 将0作为异常值赋值给未知区域对应的映射的这个步骤在ReadDataTransformtoTensor()函数中完成。
    Grid_duplicated[mappingColumnName] += 1
    # 保留对应关系。将所有去重之后的已有停留点的包含grid的dataframe保留下来。
    Grid_duplicated.to_csv(mappingSavePath)
    return Grid_duplicated
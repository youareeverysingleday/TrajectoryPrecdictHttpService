import os
import sys

sys.path.append("..")
from Common import CommonCode as cc
# from Visualization import DisplayTrajectory as dt

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
# import torchvision 
import pandas as pd

# 数据集归一化。
import sklearn.preprocessing as sk_preprocessing
import transbigdata as tbd
import numpy as np


gParameters = cc.JSONConfig('./Parameters.json')
# print(gParameters.get('gPreprocessDataSavePath'))
gGeoParameters = tbd.area_to_params(gParameters.get('gBounds'), accuracy = 1000, method='rect')

# 使用geopy来计算距离和经纬度的关系。
# import geopy
# from geopy.distance import distance as geopydistance

# transbigdata 使用经纬度时，是先使用经度longitude再用纬度latitude。
# folium 使用经纬度时，是先使用纬度latitude再使用经度 longitude。

def FillMissingLatitudeLongitude(df):
    """_summary_
    StaySeries 中存在PoIFeature部分全为空的情况。首先需要将经纬度进行填充。
    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    if pd.isna(df['longitude']) == True:
        # print(df['grid'])
        longitude, latitude = tbd.grid_to_centre([df['LONCOL'], df['LATCOL']], gGeoParameters)
        df['longitude'] = longitude
        df['latitude'] = latitude

    return df

def ReadDataTransformtoTensor(IntputSeriesPath, 
                              GridMapping,
                              PoIFeaturePath='./Data/Input/PoIFeature.csv',
                              PreprocessDataSavePath='./Data/Output/IntermediateData/', 
                              windows_length = 100,
                              drop_columns=['stime', 'etime', 'lon', 'lat', 'grid'],
                              PoIFeatureLength=16,
                              dropUsers=[]):
    """_summary_
    将数据生成多种torch专用数据格式。包括LikeLanguage、Classification、Series三种格式。

    特别注意的是：标签和样本不是相同的特征长度。因为PoI特征和轨迹特征有差异。需要通过grid来转换这种特征。
    Args:
        IntputSeriesPath (str): 输入的停留点时间序列的保存路径。是所有用户的停留点保存为一个文件的路径。
        PoIFeature (str, optional): 传入的PoI特征存储路径. Defaults to './Data/Input/PoIFeature.csv'.
        PreprocessDataSavePath (str, optional): _description_. Defaults to './Data/Output/IntermediateData/'.
        windows_length (int, optional): _description_. Defaults to 100.
        drop_columns (list, optional): 需要丢弃的列。不再丢弃经纬度，因为经纬度表示了两个相同统计特征之外的距离。注意'lon', 'lat' 这两列是轨迹的真实经纬度。
                                        最后还有两列是'longitude', 'latitude' 是轨迹对应格栅的中心经纬度。这里为了减少不必要的误差，使用grid的经纬度。
                                        . Defaults to ['stime', 'etime', 'lon', 'lat'].
        PoIFeatureLength (int, optional): PoI特征的长度，用于生存标签数据. Defaults to 16.
        dropUsers (list, optional): 需要作为测试集的用户ID. Defaults to [].

    Returns:
        GridNumberofCategory (int): 返回输入数据中包含的grid种类数量.
        Scaler (sklearn.preprocessing._data.MinMaxScaler): 返回归一化函数对象，用于逆归一化使用. 
        d_model (int): 返回特征的维度。可以理解为单词的嵌入向量长度。
    """
    
    StaySeries = pd.read_csv(IntputSeriesPath, index_col=0)
    
    # 如果需要作为测试数据集的用于ID，那么就需要将其删除。
    if len(dropUsers) != 0:
        # 注意dataframe之前的~表示的是取反操作。那么下面这段代码的意思就是：
        # ~StaySeries.index.isin([4, 68, 144]) 取index中不包含4, 68, 144的部分。
        StaySeries = StaySeries[~StaySeries.index.isin([4, 68, 144])]
    # 对 StaySeries 中PoIFeature为空的部分进行填充。
    # 并且对PoI统计特征中为nan的部分全部填0 。
    StaySeries = StaySeries.apply(FillMissingLatitudeLongitude, axis=1)
    StaySeries.fillna(0, inplace=True)
    # 将grid映射到一个合理的范围中。
    # 注意GridMapping中已经将0空了出来作为未知区域grid的映射值来处理。
    StaySeries = StaySeries.merge(right=GridMapping, on='grid', how='left').fillna(0)
    # 将grid_mapping 放置到原来grid所在列的位置。以方便后面的按列号进行的操作。
    TempCol = StaySeries.pop('grid_mapping')
    StaySeries.insert(9, 'grid_mapping', TempCol)
    print(StaySeries.columns)
    
    PoIFeature = pd.read_csv(PoIFeaturePath, index_col=0)
    # print(StaySeries['grid'].mean())
    # StaySeries.head(3)
    # y_series_Classification = np.array(StaySeries['grid'].copy())
    # 需要在可视化轨迹的时候展示。而且需要和输入模型的数据的形状对应起来。
    # 同时不能进行归一化。

    # 注意，这里将纬度latitude放在前面，经度longitude放在后面。
    StaySeries_grid_latlon = StaySeries[['grid_mapping', 'latitude', 'longitude']]
    # 将pd转为np，同时将形状变为（samples_num, 1）。
    StaySeries_onlygrid_np = StaySeries[['grid_mapping']].values.reshape(1, -1).squeeze()

    # 删除'stime', 'etime' 的原因是因为已经将时间作了分桶和分类处理。也就是将时间分为了年、月、日、周、小时等。
    # 删除'lon', 'lat' 的原因是已经将经纬度格栅化了。
    StaySeries_contain_grid = StaySeries.drop(labels=drop_columns, axis=1).copy()
    # print(StaySeries_contain_grid.columns)

    print('StaySeries_contain_grid nan count {}.'.format(StaySeries_contain_grid.isna().sum().sum() ))
    # 填充nan的值。因为nan值会导致模型的损失直接为nan。
    StaySeries_contain_grid.fillna(0, inplace=True)
    print('StaySeries_contain_grid after fill nan. nan count {}.'.format(StaySeries_contain_grid.isna().sum().sum() ))

    print('StaySeries_grid_latlon.shape {}, StaySeries_contain_grid.shape {}.'.format(StaySeries_grid_latlon.shape, StaySeries_contain_grid.shape))
    
    # gridSeries = StaySeries['grid']
    # 查看种类数量。
    print(StaySeries_contain_grid['grid_mapping'].nunique())
    # StaySeries.head(3)
    StaySeries_contain_columns = StaySeries_contain_grid.columns.tolist()
    StaySeries_contain_np = StaySeries_contain_grid.values
    print("StaySeries_contain_np shape {}.".format(StaySeries_contain_np.shape))
    d_model = StaySeries_contain_np.shape[1]

    # 归一化。-- 
    series_scaler = sk_preprocessing.MinMaxScaler(feature_range=(0, 1))
    # 这个数据集是提供给transformer和LSTM使用的。
    StaySeries_contain_np_scaler = series_scaler.fit_transform(StaySeries_contain_np)
    # StaySeries_contain_np_scaler = series_scaler.transform(StaySeries_contain_np)
    print("StaySeries_contain_np_scaler shape {}.".format(StaySeries_contain_np_scaler.shape))
    print("------------------------")

    # 1. 生成类似翻译的数据集。
    StayMatrix, x_series_LikeLanguage, y_series_LikeLanguage = cc.data_split_twodimension_to_matrix(StaySeries_contain_np_scaler, 
                                                                                                    windows_length=windows_length)
    # print("LikeLanguage x shape {}, y shape {}.".format(x_series_LikeLanguage.shape, y_series_LikeLanguage.shape))
    # 需要生成没有归一化的经纬度。因为需要计算距离。
    _, _, y_series_LikeLanguage_lonlat = cc.data_split_twodimension_to_matrix(StaySeries_contain_np, windows_length=windows_length)
    y_LikeLanguage_PoIFeatureLength = y_series_LikeLanguage[:,:,-PoIFeatureLength:]
    print("LikeLanguage x shape {}, y shape {} y_PoIFeatureLenght shape {}.".format(x_series_LikeLanguage.shape, 
                                                                                    y_series_LikeLanguage.shape,
                                                                                    y_LikeLanguage_PoIFeatureLength.shape))
    StayMatrix = torch.Tensor(StayMatrix)
    x_LikeLanguage = torch.Tensor(x_series_LikeLanguage)
    y_LikeLanguage = torch.Tensor(y_series_LikeLanguage)
    y_LikeLanguage_PoIFeatureLength = torch.Tensor(y_LikeLanguage_PoIFeatureLength)

    torch.save(StayMatrix, PreprocessDataSavePath + 'StayMatrix.pt')
    torch.save(x_LikeLanguage, PreprocessDataSavePath + 'x_torch_LikeLanguage.pt')
    torch.save(y_LikeLanguage, PreprocessDataSavePath + 'y_torch_LikeLanguage.pt')
    torch.save(y_LikeLanguage_PoIFeatureLength, PreprocessDataSavePath + 'y_torch_LikeLanguage_PoIFeatureLength.pt')

    # 1.1 生成只有经纬度的标签数据。
    y_LikeLanguage_latlon = y_series_LikeLanguage_lonlat[:,:,-2:]
    # 注意，这里将纬度放在前面，经度放在后面。
    y_LikeLanguage_latlon[:, :, [0, 1]] = y_LikeLanguage_latlon[:, :, [1, 0]]

    print("y_LikeLanguage_latlon shape {}.".format(y_LikeLanguage_latlon.shape))
    torch.save(y_LikeLanguage_latlon, PreprocessDataSavePath + 'y_torch_LikeLanguage_latlon.pt')
    print("y_LikeLanguage_latlon samples is {}".format(y_LikeLanguage_latlon[0, 0:2, :]))

    # 1.2 提供给混合损失函数使用的标签。
    y_LikeLanguage_traj_latlon = torch.concat((torch.Tensor(y_LikeLanguage), 
                                                     torch.Tensor(y_LikeLanguage_latlon)), dim=2)
    print("y_LikeLanguage_traj_latlon shape {}.".format(y_LikeLanguage_traj_latlon.shape))
    torch.save(y_LikeLanguage_traj_latlon, PreprocessDataSavePath + 'y_LikeLanguage_traj_latlon.pt')
    # print("y_LikeLanguage_PoIFeature_latlon samples is {}".format(y_LikeLanguage_PoIFeature_latlon[0, 0:2, :]))

    # 2. 生成类似时间序列的数据集。
    x_series, y_series = cc.data_split_twodimension(StaySeries_contain_np_scaler, windows_length=100)
    _, y_series_noscaler = cc.data_split_twodimension(StaySeries_contain_np, windows_length=100)
    y_series_latlon = y_series_noscaler[:,-2:]
    # 注意，这里将纬度放在前面，经度放在后面。
    y_series_latlon[:, [0, 1]] = y_series_latlon[:, [1, 0]]
    y_series_PoIFeatureLength = y_series[:,-gParameters.get('gPoIFeatureLength'):]
    print("y_series_latlon samples is {}".format(y_series_latlon[0:2, :]))
    print("y_series_PoIFeatureLength samples is {}".format(y_series_PoIFeatureLength[0:2, :]))
    print("series x shape {}, y shape {} y_series_latlon shape {} y_series_PoIFeatureLength shape {}.".format(x_series.shape, 
                                                                                                              y_series.shape, 
                                                                                                              y_series_latlon.shape,
                                                                                                              y_series_PoIFeatureLength.shape))
    
    x_Series = torch.Tensor(x_series)
    y_Series = torch.Tensor(y_series)
    y_series_latlon = torch.Tensor(y_series_latlon)
    y_series_PoIFeatureLength = torch.Tensor(y_series_PoIFeatureLength)
    y_series_traj_latlon = torch.concat((y_Series, y_series_latlon), dim=1)

    torch.save(x_Series, PreprocessDataSavePath + 'x_torch_Series.pt')
    torch.save(y_Series, PreprocessDataSavePath + 'y_torch_Series.pt')
    torch.save(y_series_latlon, PreprocessDataSavePath + 'y_torch_Series_latlon.pt')
    torch.save(y_series_PoIFeatureLength, PreprocessDataSavePath + 'y_torch_Series_PoIFeatureLength.pt')
    torch.save(y_series_traj_latlon, PreprocessDataSavePath + 'y_series_traj_latlon.pt')

    print('y_series_traj_latlon shape {}.'.format(y_series_traj_latlon.shape))


    # StaySeries_delete_grid = StaySeries.drop(labels=['stime', 'etime', 'lon', 'lat', 'grid'], axis=1).copy()
    # StaySeries_delete_np = StaySeries_delete_grid.values

    # 3. 生成类似分类问题的数据集。
    x_series_Classification_temp, y_series_Classification_temp = \
        cc.data_twodimension_to_threedimension_series(StaySeries_contain_np, 
                                                      delete_index=StaySeries_contain_columns.index('grid_mapping'),
                                                      windows_length=100)
    # 删除grid所对在的列。torch.Tensor(x_series_Classification)
    # x_series_Classification_temp = np.delete(x_series_Classification_temp, obj=StaySeries_contain_columns.index("grid"), axis=1)
    x_series_Classification = F.normalize(torch.Tensor(x_series_Classification_temp), dim=2)
    
    # print("Classification y shape {}.".format(y_series_temp.shape))
    # print(StaySeries_columns.index("grid"))
    # 将其中的grid列作为标签取出来。
    y_series_Classification = pd.DataFrame(y_series_Classification_temp[:,StaySeries_contain_columns.index('grid_mapping')], 
                                           columns=['grid_mapping'])
    # 获得地域的分类数量。
    # 去重之后的数据。单纯看数值太大了，而且有负值和正值。两个值。实际上类别数量只有2999个。
    GridNumberofCategory = y_series_Classification.nunique()
    
    # 这里之所以比最原始的pandas的获取的分类数少13个（原始的是2999类，现在是2986类）
    # 是因为在生成数据集的时候最后100个样本中的grid并没有统计在标签中
    print("Classification's number of category {}.".format(GridNumberofCategory))
    y_series_Classification = pd.get_dummies(y_series_Classification, columns=['grid_mapping'], 
                                             prefix_sep='_', dummy_na=False, drop_first=False).values
    # y_series_Classification.shape[0]
    # y_series_Classification = y_series_Classification.reshape(y_series_Classification.shape[0], 1, y_series_Classification.shape[1])
    # print(y_series_Classification.shape, type(y_series_Classification))
    # x_series_Classification = x_series_Classification.reshape(x_series_Classification[0], )
    
    print("Classification x shape {}, y shape {}.".format(x_series_Classification.shape, y_series_Classification.shape))
    x_Classification = torch.Tensor(x_series_Classification)
    y_Classification = torch.Tensor(y_series_Classification)

    torch.save(x_Classification, PreprocessDataSavePath + 'x_torch_Classification.pt')
    torch.save(y_Classification, PreprocessDataSavePath + 'y_torch_Classification.pt')
    
    # 输入CNN的模型样本x需要有channel这一个维度，所以需要升维。
    x_CNN_Classification = torch.Tensor(x_series_Classification).unsqueeze(1)
    y_CNN_Classification = torch.Tensor(y_series_Classification)
    print("CNN Classification x shape {}, y shape {}.".format(x_CNN_Classification.shape, y_CNN_Classification.shape))
    
    torch.save(x_CNN_Classification, PreprocessDataSavePath + 'x_torch_CNN_Classification.pt')
    torch.save(y_CNN_Classification, PreprocessDataSavePath + 'y_torch_CNN_Classification.pt')
    
    # 同样的输入CNN的模型样本x需要有channel这一个维度，所以需要升维。
    # 与上面输入CNN不同的是这里用于损失函数是交叉熵的CNN。
    x_CNN_Series = torch.Tensor(x_series).unsqueeze(1)
    y_CNN_Series = torch.Tensor(y_series)
    print("series x CNN shape {}, y CNN shape {}.".format(x_CNN_Series.shape, y_CNN_Series.shape))

    torch.save(x_CNN_Series, PreprocessDataSavePath + 'x_torch_CNN_Series.pt')
    torch.save(y_CNN_Series, PreprocessDataSavePath + 'y_torch_CNN_Series.pt')

    # 4. 生成 grid 矩阵。
    # 只有gird。
    # # 主要用于transformer中topN推荐。可以最终判断准确率。
    # grid_matrix, grid_x_LikeLanguage, grid_y_LikeLanguage = cc.data_split_onedimension(gridSeries.values, windows_length=100)
    # grid_matrix = torch.Tensor(grid_matrix)
    # grid_x_LikeLanguage = torch.Tensor(grid_x_LikeLanguage)
    # grid_y_LikeLanguage = torch.Tensor(grid_y_LikeLanguage)

    # print("grid_x_LikeLanguage x shape {}, grid_y_LikeLanguage shape {}.".format(grid_x_LikeLanguage.shape, grid_y_LikeLanguage.shape))

    # torch.save(grid_matrix, PreprocessDataSavePath + 'grid_matrix.pt')
    # torch.save(grid_x_LikeLanguage, PreprocessDataSavePath + 'grid_x_LikeLanguage.pt')
    # torch.save(grid_y_LikeLanguage, PreprocessDataSavePath + 'grid_y_LikeLanguage.pt')

    # 4.2 将 grid, longitude, latitude 保存下来。并且需要和生成的类似翻译数据集的前两维形状相同。 
    # 包含了grid, longitude, latitude 三列，避免之后在可视化轨迹时再次运算一次。StaySeries_contain_np_scaler
    grid_matrix, grid_x_LikeLanguage, grid_y_LikeLanguage = cc.data_split_twodimension_to_matrix(StaySeries_grid_latlon, windows_length=windows_length)

    grid_matrix = torch.Tensor(grid_matrix)
    grid_x_LikeLanguage = torch.Tensor(grid_x_LikeLanguage)
    grid_y_LikeLanguage = torch.Tensor(grid_y_LikeLanguage)

    print("grid_x_LikeLanguage x shape {}, grid_y_LikeLanguage shape {}.".format(grid_x_LikeLanguage.shape, grid_y_LikeLanguage.shape))

    torch.save(grid_matrix, PreprocessDataSavePath + 'grid_matrix.pt')
    torch.save(grid_x_LikeLanguage, PreprocessDataSavePath + 'grid_x_LikeLanguage.pt')
    torch.save(grid_y_LikeLanguage, PreprocessDataSavePath + 'grid_y_LikeLanguage.pt')

    # 4.3 只使用grid。并且输出形状为x = (samples_number, windows_length) 和 y = (samples_number, 1)。
    x_onlygrid = []
    y_onlygrid = []

    for i in range(len(StaySeries_onlygrid_np) - windows_length - 1 + 1):
        sample = StaySeries_onlygrid_np[i:i + windows_length]  # 提取样本
        label = StaySeries_onlygrid_np[i + windows_length:i + windows_length + 1]  # 提取标签
        x_onlygrid.append(sample)
        y_onlygrid.append(label)

    # 将样本和标签转换为 DataFrame
    x_onlygrid = torch.tensor(x_onlygrid)
    y_onlygrid = torch.tensor(y_onlygrid)

    torch.save(x_onlygrid, PreprocessDataSavePath + 'x_onlygrid.pt')
    torch.save(y_onlygrid, PreprocessDataSavePath + 'y_onlygrid.pt')

    return int(GridNumberofCategory), series_scaler, d_model


# def GetDataLoader(x_path, y_path, batch_size=16, train_size=0.8):
#     x = torch.load(x_path)
#     if not torch.is_tensor(x):
#         x = torch.from_numpy(x)

#     y = torch.load(y_path)
#     if not torch.is_tensor(y):
#         y = torch.from_numpy(y)
#     print("file path {} ,x shape{}, y shape {}.".format(x_path, x.shape, y.shape))
#     dataset = TensorDataset(x, y)

#     # 将数据集分为训练集和测试集
#     # train_dataset, test_dataset 
#     # trainLoader, testLodaer
#     train_dataset, test_dataset = random_split(dataset, 
#                                            lengths=[int(train_size * len(dataset)), 
#                                                     len(dataset) - int(train_size * len(dataset))],
#                                            generator=torch.Generator().manual_seed(0))
    
#     trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
#     testLodaer = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
#     return trainLoader, testLodaer


if __name__ == '__main__':
    # 训练设备。
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)

    gParameters = cc.JSONConfig('./Parameters.json')
    # EPOCHS = 100
    gBounds = gParameters.get('gBounds')
    gGeoParameters = tbd.area_to_params(gBounds, accuracy = 1000, method='rect')

    PoIFeature = pd.read_csv('./Data/Input/PoIFeature.csv', index_col=0)
    gParameters.set('gPoIFeatureLength', PoIFeature.shape[1])
    print(gParameters.get('gPoIFeatureLength'))

    # 先生成将grid映射到合理范围的映射表。
    AllGridMapping = cc.GenerateAllGridMapping(Bounds=gBounds)

    # 将这3个用户（4, 68, 144）的数据从训练数据集中删除作为测试数据。
    # 将映射表传入。
    ClassificationCounter, Scaler, temp_d_model= ReadDataTransformtoTensor(GridMapping=AllGridMapping,
                                                                           IntputSeriesPath=gParameters.get('gFeatureSeriesInputPath'), 
                                                                        PoIFeatureLength=gParameters.get('gPoIFeatureLength'),
                                                                        dropUsers=[4, 68, 144])
    gParameters.set('g_d_model', temp_d_model)

    print(ClassificationCounter, gParameters.get('g_d_model'))

    # 单个用户数据生成。
    ClassificationCounter_4, Scaler_4, _ = ReadDataTransformtoTensor(GridMapping=AllGridMapping,
                                                                     IntputSeriesPath='./Data/Input/Stay/004.csv', 
                                                                     PreprocessDataSavePath='./Data/Output/IntermediateData/SingleUserStay/')
    print(ClassificationCounter_4, Scaler_4)

    print('Data preparation has completed.')

    # 在数据预处理中没必要加载到loader中。在算法部分加载到loader中即可。
    # # torch tensor 的保存路径。
    # PreprocessDataSavePath = './Data/Output/IntermediateData/'
    # # 将所有的数据都读为torch的loader。
    # # 轨迹数据的特征来作为标签。交叉熵作为损失。
    # if os.path.exists(PreprocessDataSavePath + 'x_torch_LikeLanguage.pt'):
    #     trainLoader_LikeLanguage, testLodaer_LikeLanguage = cc.GetDataLoader(PreprocessDataSavePath + 'x_torch_LikeLanguage.pt', 
    #                                                                     PreprocessDataSavePath + 'y_torch_LikeLanguage.pt',
    #                                                                     batch_size=gParameters.get('BATCH_SIZE'), train_size=0.8)
    # else:
    #     print("x_torch_LikeLanguage.pt is not exist.")

    # # PoI特征来作为标签。交叉熵作为损失。
    # if os.path.exists(PreprocessDataSavePath + 'y_torch_LikeLanguage_PoIFeatureLength.pt'):
    #     trainLoader_LikeLanguage_PoIFeatureLength, testLodaer_LikeLanguage_PoIFeatureLength = cc.GetDataLoader(PreprocessDataSavePath + 'x_torch_LikeLanguage.pt', 
    #                                                                     PreprocessDataSavePath + 'y_torch_LikeLanguage_PoIFeatureLength.pt',
    #                                                                     batch_size=gParameters.get('BATCH_SIZE'), train_size=0.8)
    # else:
    #     print("y_torch_LikeLanguage_PoIFeatureLength.pt is not exist.")


    # # 距离作为损失函数时的数据。
    # if os.path.exists(PreprocessDataSavePath + 'y_torch_LikeLanguage_latlon.pt'):
    #     trainLoader_LikeLanguage_latlon, testLodaer_LikeLanguage_latlon = cc.GetDataLoader(PreprocessDataSavePath + 'x_torch_LikeLanguage.pt', 
    #                                                                     PreprocessDataSavePath + 'y_torch_LikeLanguage_latlon.pt',
    #                                                                     batch_size=gParameters.get('BATCH_SIZE'), train_size=0.8)
    # else:
    #     print("y_torch_LikeLanguage_latlon.pt is not exist.")

    # # poi和距离混合作为损失时。
    # if os.path.exists(PreprocessDataSavePath + 'y_LikeLanguage_PoIFeature_latlon.pt'):
    #     trainLoader_LikeLanguage_PoIFeature_latlon, testLodaer_LikeLanguage_PoIFeature_latlon = cc.GetDataLoader(PreprocessDataSavePath + 'x_torch_LikeLanguage.pt', 
    #                                                                     PreprocessDataSavePath + 'y_LikeLanguage_PoIFeature_latlon.pt',
    #                                                                     batch_size=gParameters.get('BATCH_SIZE'), train_size=0.8)
    # else:
    #     print("y_LikeLanguage_PoIFeature_latlon.pt is not exist.")


    # if os.path.exists(PreprocessDataSavePath + 'x_torch_Classification.pt'):
    #     trainLoader_Classification, testLodaer_Classification = cc.GetDataLoader(PreprocessDataSavePath + 'x_torch_Classification.pt', 
    #                                                                         PreprocessDataSavePath + 'y_torch_Classification.pt',
    #                                                                         batch_size=gParameters.get('BATCH_SIZE'), train_size=0.8)
    # else:
    #     print("x_torch_Classification.pt is not exist.")
        
    # if os.path.exists(PreprocessDataSavePath + 'x_torch_CNN_Classification.pt'):
    #     trainLoader_CNN_Classification, testLodaer_CNN_Classification = cc.GetDataLoader(PreprocessDataSavePath + 'x_torch_CNN_Classification.pt', 
    #                                                                         PreprocessDataSavePath + 'y_torch_CNN_Classification.pt',
    #                                                                         batch_size=gParameters.get('BATCH_SIZE'), train_size=0.8)
    # else:
    #     print("x_torch_CNN_Classification.pt is not exist.")

    # # 序列类的数据。
    # if os.path.exists(PreprocessDataSavePath + 'x_torch_Series.pt'):
    #     trainLoader_Series, testLodaer_Series = cc.GetDataLoader(PreprocessDataSavePath + 'x_torch_Series.pt', 
    #                                                         PreprocessDataSavePath + 'y_torch_Series.pt',
    #                                                         batch_size=gParameters.get('BATCH_SIZE'), train_size=0.8)
    # else:
    #     print("x_torch_Series.pt is not exist.")

    # if os.path.exists(PreprocessDataSavePath + 'y_torch_Series_PoIFeatureLength.pt'):
    #     trainLoader_Series_PoIFeatureLength, testLodaer_Series_PoIFeatureLength = cc.GetDataLoader(PreprocessDataSavePath + 'x_torch_Series.pt', 
    #                                                         PreprocessDataSavePath + 'y_torch_Series_PoIFeatureLength.pt',
    #                                                         batch_size=gParameters.get('BATCH_SIZE'), train_size=0.8)
    # else:
    #     print("y_torch_Series_PoIFeatureLength.pt is not exist.")

    # if os.path.exists(PreprocessDataSavePath + 'y_torch_Series_latlon.pt'):
    #     trainLoader_Series_latlon, testLodaer_Series_latlon = cc.GetDataLoader(PreprocessDataSavePath + 'x_torch_Series.pt', 
    #                                                         PreprocessDataSavePath + 'y_torch_Series_latlon.pt',
    #                                                         batch_size=gParameters.get('BATCH_SIZE'), train_size=0.8)
    # else:
    #     print("y_torch_Series_latlon.pt is not exist.")

    # # 输入CNN的序列数据。和输入MLP的数据不同点在于样本数据升维了，标签都是一样的。
    # if os.path.exists(PreprocessDataSavePath + 'x_torch_CNN_Series.pt'):
    #     trainLoader_CNN_Series, testLodaer_CNN_Series = cc.GetDataLoader(PreprocessDataSavePath + 'x_torch_CNN_Series.pt', 
    #                                                         PreprocessDataSavePath + 'y_torch_CNN_Series.pt',
    #                                                         batch_size=gParameters.get('BATCH_SIZE'), train_size=0.8)
    # else:
    #     print("x_CNN_torch_Series.pt is not exist.")


    # if os.path.exists(PreprocessDataSavePath + 'y_torch_Series_PoIFeatureLength.pt'):
    #     trainLoader_CNN_Series_PoIFeatureLength, testLodaer_CNN_Series_PoIFeatureLength = cc.GetDataLoader(PreprocessDataSavePath + 'x_torch_CNN_Series.pt', 
    #                                                         PreprocessDataSavePath + 'y_torch_Series_PoIFeatureLength.pt',
    #                                                         batch_size=gParameters.get('BATCH_SIZE'), train_size=0.8)
    # else:
    #     print("y_torch_Series_PoIFeatureLength.pt is not exist.")

    # if os.path.exists(PreprocessDataSavePath + 'y_torch_Series_latlon.pt'):
    #     trainLoader_CNN_Series_latlon, testLodaer_CNN_Series_latlon = cc.GetDataLoader(PreprocessDataSavePath + 'x_torch_CNN_Series.pt', 
    #                                                         PreprocessDataSavePath + 'y_torch_Series_latlon.pt',
    #                                                         batch_size=gParameters.get('BATCH_SIZE'), train_size=0.8)
    # else:
    #     print("y_torch_Series_latlon.pt is not exist.")
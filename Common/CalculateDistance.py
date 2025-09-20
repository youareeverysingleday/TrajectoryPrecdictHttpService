
# Author: youareeverysingleday
# contact: @
# CreationTime: 2020/6/18 16:17
# software: VSCode
# LastEditTime: 2024/03/01 18:28
# LastEditors: youareeverysingleday
# Description: multiple calculation methods of distance.

import numpy as np
import math

def L1Distance(vector_1, vector_2):
    """_summary_
    计算L1范数。要求两个向量长度相等。
    计算方法：
        1. 两个向量中每个对应位置元素差值的绝对值；
        2. 求1中所有值之和。
        $d_1 (I_1, I_2) = \sum \limit_{j=0}^n \vert i_{1j} - i_{2_j}\vert$
    Args:
        vector_1 (pandas.Series): 输入的第一个向量。
        vector_2 (pandas.Series): 输入的第二个向量。

    Returns:
        L1 (float): 返回L1距离。
    """
    L1 = 0.0
    for e1, e2 in zip(vector_1, vector_2):
        L1 += abs(e1 - e2)
    return L1

def L2Distance(vector_1, vector_2):
    """_summary_
    计算L2范数。要求两个向量长度相等。
    计算方法：
        1. 两个向量中每个对应位置元素差值的平方；
        2. 求1中所有值之和。
        3. 求2中值的平方。
        $d_2 (I_1, I_2) = \sqrt{} \sum \limit_{j=0}^n (i_{1j} - i_{2_j})^2$
    Args:
        vector_1 (pandas.Series): 输入的第一个向量。
        vector_2 (pandas.Series): 输入的第二个向量。

    Returns:
        L2 (float): 返回L2距离。
    """
    L2 = 0.0
    for e1, e2 in zip(vector_1, vector_2):
        L2 += (e1 - e2) ** 2
    L2 = math.sqrt(L2)
    return L2

def Cosine(vector_1, vector_2):
    """_summary_
    求两个向量之间的余弦相似度。要求vector_1、 vector_2长度相同。
    Args:
        vector_1 (pandas.Series): _description_
        vector_2 (pandas.Series): _description_

    Returns:
        cosine (float): 返回余弦相似度。
    """
    # 使用矩阵点乘。计算两个向量的点乘。
    sum = np.dot(np.mat(vector_1),  np.mat(vector_2).T)
    # sum = np.dot(np.array(vector_1),  np.array(vector_2).T)
    # 计算向量的范数。
    denom = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
    # print(sum, np.linalg.norm(vector_1) , np.linalg.norm(vector_2))
    return sum/denom


def VectorKLDivergence(vector1, vector2):
    """_summary_
    求两个向量之间的KL散度。要求两个向量的形状相同。
    Args:
        vector1 (numpy.ndarray): 第一个向量。
        vector2 (numpy.ndarray): 第二个向量。

    Returns:
        float: 返回散度值。
    """
    if vector1.shape != vector2.shape:
        print('the shape between two vectors is different.')
        return 0.0
    
    p1 = 1.0 *vector1 /np.sum(vector1)
    p2 = 1.0 *vector2 /np.sum(vector2)

    # 元素之间的乘法。
    return np.sum(np.multiply(p1, (np.log(p1) - np.log(p2))))

def MatrixKLDivergence_SameShape(matrix1, matrix2):
    """_summary_
    两个矩阵所有对应行上的KL散度。
    要求两个矩阵的形状完全相同。
    只能计算按行求的散度。
    Args:
        matrix1 (numpy.ndarray): 第一个矩阵。
        matrix2 (numpy.ndarray): 第二个矩阵。

    Returns:
        numpy.ndarray: 返回散度值的行向量。
    """
    if matrix1.shape[1] != matrix2.shape[1]:
        print('the second dimension of shape between two matrices is different.')
        return 0.0

    p1 = 1.0 * matrix1 / np.sum(matrix1, axis=1)[:,None]
    p2 = 1.0 * matrix2 / np.sum(matrix2, axis=1)[:,None]

    return np.sum(np.multiply(p1, (np.log(p1) - np.log(p2))), axis=1)


def MatrixKLDivergence_SameSecondDimension(matrix1, matrix2):
    """_summary_
    求第一个矩阵的每一行和第二个矩阵的每一行的KL散度。
    要求两个矩阵的第二个维度的长度相同。
    只能计算按行求的散度。
    经过scipy.stats.entropy()的验证是对的。
    Args:
        matrix1 (numpy.ndarray): 第一个矩阵。
        matrix2 (numpy.ndarray): 第二个矩阵。

    Returns:
        numpy.ndarray: 返回散度值的行向量。
    """
    if matrix1.shape[1] != matrix2.shape[1]:
        print('the second dimension of shape between two matrices is different.')
        return 0.0

    p1 = 1.0 * matrix1 / np.sum(matrix1, axis=1)[:,None]
    log1 = np.log(p1)
    p2 = 1.0 * matrix2 / np.sum(matrix2, axis=1)[:,None]
    log2 = np.log(p2)

    log2Extension = np.empty(shape=(0, matrix1.shape[1]))
    for row in log2:
        # 将log2展开。log2的每行向量都变为log1的第一个维度的大小。
        # 然后log1减去展开之后的log2。
        temp = log1 - (np.tile(row, (matrix1.shape[0], 1)))
        log2Extension = np.vstack((log2Extension, temp))

    # 计算的结果修改形状为3维。
    log2Extension = log2Extension.reshape(matrix2.shape[0], matrix1.shape[0], matrix1.shape[1])
    # 然后计算出最后的KL散度。
    KLDivergence = np.sum(np.multiply(p1, log2Extension), axis=2)
    # print(KLDivergence.shape)
    # 最后需要转置一次是因为需要输入矩阵的逻辑相同。也就是输出矩阵的第一个维度是输入矩阵的第一个维度。
    return KLDivergence.T

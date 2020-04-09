# coding: UTF-8
# 不属于同一量纲：即特征的规格不一样，不能够放在一起比较。无量纲化可以解决这一问题。
"""
# 与MinMax相比适合整体都不太规整，方差较大的场景，符合正态分布

# 公式为：(X-X_mean)/X_std 计算时对每个属性/每列分别进行.
# 将数据按其属性(按列进行)减去其均值，然后除以其方差。最后得到的结果是，对每个属性/每列来说所有数据都聚集在0附近，方差值为1。
--------------------------------------------------------
# sklearn.preprocessing.scale(X, axis=0, with_mean=True,with_std=True,copy=True)
# 参数解释：
    # X：数组或者矩阵
    # axis：int类型，初始值为0，axis用来计算均值 means 和标准方差 standard deviations.
        如果是0，则单独的标准化每个特征（列），如果是1，则标准化每个观测样本（行）。
    # with_mean: boolean类型，默认为True，表示将数据均值规范到0
    # with_std: boolean类型，默认为True，表示将数据方差规范到1
    # 根据参数的不同，可以沿任意轴标准化数据集。
"""

"""----------------
手工实现
-------------------"""
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
data = iris.data
m_data = np.mean(data, axis=0)
print(m_data, m_data)
std_data = np.std(data, axis=0)
print(std_data.shape, std_data)
n_data = (data - m_data) / std_data
print(n_data.shape, n_data[0])

"""----------
sklearn 实现
------------"""

from sklearn.preprocessing import StandardScaler

data = [[1, 10], [2, 20], [3, 30], [4, 40]]
print(data)
# 创建
scaler = StandardScaler()
"""
# fit数据，相当于计算均值方差
# StandardScaler(copy=True, with_mean=True, with_std=True)
"""
scaler.fit(data)
""" 均值及方差 """
print(scaler.mean_)
""" 转化数据 """
result = scaler.transform(data)
# result = scaler.fit_transform(data)
print(result)

'''-----------
pandas 标准化
--------------'''
import pandas as pd

df = pd.DataFrame(data, columns=['one', 'tow'])
df['tow'] = (df['tow'] - df['tow'].mean()) / df['tow'].std()
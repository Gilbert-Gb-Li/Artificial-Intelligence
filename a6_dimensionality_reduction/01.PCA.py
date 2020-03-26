#!/usr/bin/python
# -*- coding: utf-8 -*-
# cangye@hotmail.com
"""
PCA 主成分分析
"""
from sklearn.datasets import make_circles

print(__doc__)

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.decomposition import PCA, FastICA, KernelPCA

mean = [0, 0]  # 平均值
cov = [[1, 0.9], [0.9, 1]]  # 协方差

x, y = np.random.multivariate_normal(mean, cov, 1000).T
x = np.reshape(x, [-1, 1])
y = np.reshape(y, [-1, 1])
X = np.concatenate([x, y], axis=1)
label = np.zeros_like(x[:, 0])
label[x[:, 0] > 1] = 1
label[x[:, 0] < -1] = 1

"""-----------------------------
1.
参数：
    n_components：
    # 整形则保留的列数量
    # 小数表示保留百分之多少的信息
    # PCA(n_components=3) 等价于 fit_transform(X)[:, :3] 【只取前三个特征】
属性：
    components_：返回具有最大方差的成分。
    explained_variance_ratio_：返回 所保留的n个成分各自的方差百分比。
    n_components_：返回所保留的成分个数n。
    mean_：
    noise_variance_：
    
# fit -> 得到特征矩阵
# transform(X) -> 得到降维后的矩阵

# PCA无法完成非线性问，对make_circles数据集效果不好
# X, y = make_circles(n_samples=400, factor=.3, noise=.05)
---------------------------------"""

pca = PCA(n_components=0.99)

"""---------------
2.
独立成分分析, ICA
------------------"""
# pca = FastICA(n_components=0.99)

"""------------------
3.
非线性可分降维
主成分分析(Principal Components Analysis, PCA)适用于数据的线性降维。
而核主成分分析(Kernel PCA, KPCA)可实现数据的非线性降维，用于处理线性不可分的数据集
升维
参数：
    n_components: int，要返回的主要组件的数量
    kernel：
    gamma: float， RBF核的调优参数
---------------------"""
# pca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)

''' # fit and transform '''
X_pca = pca.fit_transform(X)
print(X.shape, X_pca.shape)


mpl.style.use('fivethirtyeight')
fig = plt.figure()
ax = fig.add_subplot(211)
ax.scatter(X[:, 0], X[:, 1], c=label)
ax.axis("equal")
ax = fig.add_subplot(212)
ax.scatter(X_pca[:, 0] * pca.explained_variance_[0], X_pca[:, 1] * pca.explained_variance_[1], c=label)
ax.axis("equal")
plt.show()

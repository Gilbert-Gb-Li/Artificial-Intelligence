#!/usr/bin/python
# -*- coding: utf-8 -*-
# cangye@hotmail.com
"""
==============================
Bootstrap aggregating(bagging)
==============================
随机森林算法属于ensemble算法的bagging分支
其核心思想就是将一些方差比较大(比较容易过拟合)的分类器进行组合
用随机的方式消除方差。
"""
print(__doc__)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons, make_circles, make_classification, load_iris

'''# 训练数据'''
# iris = load_iris()
# y = iris['target']
# X = iris['data']
# p_y = pd.DataFrame(y)
# p_X = pd.DataFrame(X)
# print(p_X.describe())

# X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
X, y = make_moons(noise=0.1, random_state=1)

"""
随机森林分类器，无需做任何特征工程
n_estimators : integer, optional (default=10)
    随机森林中树的数量；
criterion : string, optional (default=”gini”)
    gini系数，entropy熵；
max_depth : integer or None, optional (default=None)
    树的最大深度， 一般为15或者更高；
min_samples_split：int, float, optional (default=2)
    最小分裂样本数；
min_samples_leaf : int, float, optional (default=1)
    最小叶子节点样本数；
oob_score : bool (default=False)
    是否使用袋外（out-of-bag）样本估计准确度；
    obb error其实就是随机森林里面某棵树，
        用来构建它的时候可能有n个数据没有用到，然后我们用这n个数据测一遍这棵树，然后obb_error = 被分类错误数 / 总数
    1）对每个样本，计算它作为oob样本的树对它的分类情况（约1/3的树）；
    2）然后以简单多数投票作为该样本的分类结果；
    3）最后用误分个数占样本总数的比率作为随机森林的oob误分率。
n_jobs : integer, optional (default=1)
    并行job数，-1 代表全部；
random_state : int, RandomState instance or None, optional (default=None)
    随机数种子；
warm_start : bool, optional (default=False)
    如果设置为True，在之前的模型基础上预测并添加模型，否则，建立一个全新的森林；
class_weight : dict, list of dicts, “balanced”,
    “balanced” 模式自动调整权重，每类的权重为 n_samples / (n_classes * np.bincount(y))，即类别数的倒数除以每类样本数的占比。
"""
rdf = RandomForestClassifier(n_estimators=20, max_depth=8, oob_score=True)

# 训练过程
rdf.fit(X, y)
# rdf.fit(p_X, p_y)

# 绘图库引入
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# 调整图片风格
mpl.style.use('fivethirtyeight')
# 定义xy网格，用于绘制等值线图
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
# 预测可能性
Z = rdf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.8)
# 绘制散点图
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("RandomForest")
plt.axis("equal")
plt.show()

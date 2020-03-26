#!/usr/bin/python
# -*- coding: utf-8 -*-
# cangye@hotmail.com
"""
=====================
高斯环境的贝叶斯分类器
=====================

"""
from func_model.visual._20_plot_ge_2d import contourf
import numpy as np
print(__doc__)

from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_moons, make_circles, make_classification

'''引入训练数据'''
X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
# X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
#                            random_state=1, n_clusters_per_class=1)
# print(X[0], y[0:3])

'''-----------------------
在训练数据协方差相同的情况下
高斯分类器等效于线性分类器
文章：https://zhuanlan.zhihu.com/p/30824582
---------------------------'''
# X1 = np.random.normal(size=[600, 2])
# X2 = np.random.random([600, 2])
# # 方差均衡，使得两个类方差均为1
# X1 = X1 / np.std(X1)
# X2 = X2 / np.std(X2)
# y = np.concatenate([np.zeros_like(X1[:, 0]), np.ones_like(X2[:, 0])], axis=0)
# X = np.concatenate([X1, X2], axis=0)


'''定义高斯分类器类'''
gnb = GaussianNB()

'''训练过程'''
gnb.fit(X, y)

'''预测'''
# p_y = gnb.predict([[10, 2]])
# print(p_y)

# 绘图
contourf(gnb, X, y)


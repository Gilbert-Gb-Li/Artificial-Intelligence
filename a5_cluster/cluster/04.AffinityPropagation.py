#!/usr/bin/python
# -*- coding: utf-8 -*-
# cangye@hotmail.com
"""
AP聚类算法是基于数据点间的"信息传递"的一种聚类算法。
与k-均值算法或k中心点算法不同，AP算法不需要在运行算法之前确定聚类的个数。
AP算法寻找的"examplars"即聚类中心点是数据集合中实际存在的点，作为每类的代表。
"""
from func_model.visual.v_simple import contourf

print(__doc__)

from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs
from sklearn.cluster import AffinityPropagation
import numpy as np

# 1. 生产数据
# 1.1 环形数据
X, y = make_circles(noise=0.1, factor=0.1, random_state=1)

# 1.2 聚类数据
# centers = [[0, 1], [-1, -1], [1, -1]]
# X, y = make_blobs(n_samples=1500, random_state=170)
# trs = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
# X = np.dot(X, trs)

"""
"""
clt = AffinityPropagation(damping=0.5)
clt.fit(X)

contourf(clt, X, y, False)


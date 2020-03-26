# -- coding: utf-8 --
"""
逻辑回归鸢尾花多分类
"""
# from func_model.visual.v_simple import v_pcolormesh

print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

iris = datasets.load_iris()
X = iris.data[:, :2]
# X[0]: [5.1 3.5 1.4 0.2]
Y = iris.target
# Y[10]: [0 0 0 0 0 0 0 0 0 0]

logreg = linear_model.LogisticRegression(C=1e-5)

logreg.fit(X, Y)
# print("coef")
# print(logreg.coef_)

# v_pcolormesh(logreg, X, Y)


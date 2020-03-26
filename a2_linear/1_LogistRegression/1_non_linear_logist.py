# -- coding: utf-8 --
"""
正常逻辑分类无法完成非线性分类
"""
from sklearn.datasets import make_circles

print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
print(X[:, 0].shape, y.shape)
# logreg = linear_model.LogisticRegression(C=1e-5)
#
# logreg.fit(X, y)
# y_pred = logreg.predict(X)
# plt.scatter(x=X[:, 0], y=X[:, 1], c=y_pred, label='test data')


a = [1, 2, 3, 4] # y 是 a的值，x是各个元素的索引
b = [5, 6, 9, 8]
X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
# plt.plot(a, b)
plt.plot(X[:, 0], y)
plt.show()

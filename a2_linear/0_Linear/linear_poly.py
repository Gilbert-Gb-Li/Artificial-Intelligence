"""
多项式线性回归
-------------
只改变样本的指数或者生成交叉特征
是可以完成线性回归转换到非线性回归的
"""
print(__doc__)

import numpy as np
X = np.array([258.0, 270.0, 294.0, 320.0,
              342.0, 368.0, 396.0, 446.0,
              480.0, 586.0])
# shape: (10,1)
# X[0]: [258.]
X = X[:, np.newaxis]

X_origin = np.linspace(250, 600, 10)[:, np.newaxis]  # shape: (10,1)
y = np.array([236.4, 234.4, 252.8,
              298.6, 314.2, 342.2,
              360.8, 368.0, 391.2, 390.8])

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
pr = LinearRegression()

lr.fit(X, y)
y_lr = lr.predict(X_origin)

from sklearn.preprocessing import PolynomialFeatures

'''-------------------
# 将样本进行多项式变换
# 生成高次及交叉特征样本
----------------------'''
quadratic = PolynomialFeatures(degree=2)  # 二项式
# shape: (10,1)
# X_quad[0]: [1.00000e+00 2.58000e+02 6.65640e+04]
X_quad = quadratic.fit_transform(X)
X_new = quadratic.fit_transform(X_origin)
pr.fit(X_quad, y)
y_pr = pr.predict(X_new)

'''------
# 图像打印
---------'''
import matplotlib.pyplot as plt
plt.scatter(X, y, label='traing data')
plt.plot(X_origin, y_lr, label='linear fit', linestyle='--')
plt.plot(X_origin, y_pr, label='quadratic fit')
plt.legend(loc='upper left')
plt.show()

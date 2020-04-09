#coding: UTF-8
"""
多项式特征组合
特征与特征相乘，例：
有 a、b 两个特征，那么它的 2 次多项式的次数为 [1,a,b,a^2,ab,b^2]
参数：
    degree, 多项式阶数，默认为2
    interaction_only 的意思是，得到的组合特征只有相乘的项，没有平方项。
"""
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

X = np.arange(9).reshape(3, 3)
print(X)

poly = PolynomialFeatures(2)
print(poly.fit_transform(X))

poly = PolynomialFeatures(interaction_only=True)
print(poly.fit_transform(X))
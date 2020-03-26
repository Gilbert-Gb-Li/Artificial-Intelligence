#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
=====================
SVM算法
SVM算法时间复杂度取决于支持向量个数，虽然结果上与单层神经网络(逻辑回归)类似，但是实现原理并不相同
=====================
基本用法
"""
print(__doc__)

from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles, make_classification

# 引入训练数据
X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
# X, y = make_moons(noise=0.3, random_state=0)
# X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
#                            random_state=1, n_clusters_per_class=1, class_sep=1)
"""
SVC　class sklearn.svm.
SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,
    tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
    decision_function_shape='ovr', random_state=None)　
    
    C: 松弛因子，C值越大可能出现的分类错误越小，容易过拟合；
    gamma: gamma 越大，支持向量越少，gamma 越小，支持向量越多
        随着gamma增大，细节越丰富，这样注意过拟合问题
    https://zhuanlan.zhihu.com/p/86844493
　　kernel: 核函数，参数选择有RBF, Linear, Poly, Sigmoid， precomputed或者自定义一个核函数, 
    默认的是"RBF",
    
　　degree: 当指定kernel为'poly'时，表示选择的多项式的最高次数，默认为三次多项式；
　　coef0: 核函数常数值(y=kx+b中的b值)，只有‘poly’和‘sigmoid’核函数有，默认值是0。
　　tol: 残差收敛条件，默认是0.0001，即容忍1000分类里出现一个错误，与LR中的一致；误差项达到指定值时则停止训练。
　　class_weight :  {dict, ‘balanced’}，字典类型或者'balance'字符串。
　　max_iter: 最大迭代次数，默认是-1，即没有限制。这个是硬限制，
    它的优先级要高于tol参数，不论训练的标准和精度达到要求没有，都要停止训练。
　　decision_function_shape: 多分类参数，‘ovo’，为one v one，一对一，
        ‘ovr’ 一对多，为one v rest，默认是ovr，因为此种效果要比oro略好一点。【不建议多分类】
"""

lsvm = SVC(kernel='rbf', C=1.0, gamma=1)
# 训练过程
lsvm.fit(X, y)



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

'''预测可能性'''
Z = lsvm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=.8)
# 绘制散点图
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
sv = lsvm.support_vectors_
print(sv.shape)
print(sv[0])
plt.scatter(sv[:, 0], sv[:, 1], s=20, marker='^', color="#000099")
plt.title("SVM")
plt.axis("equal")
plt.show()

print(__doc__)
# Author: Noel Dawe <noel.dawe@gmail.com>
#
# License: BSD 3 clause
# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

# Create the dataset
rng = np.random.RandomState(1)
X = np.linspace(0, 6, 100)[:, np.newaxis]
# [[1], [2], ]
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])
print(X.shape)
print(y.shape)

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=4)

# 计算300棵树
'''----------
调参参数：
    n_estimators: 学习器个数，默认是100，常常将n_estimators和下面介绍的参数learning_rate一起考虑 => 300
    learning_rate: 每个弱学习器的权重缩减系数ν，也称作步长，fk(x)=fk−1(x)+νhk(x)，ν的取值范围为0<ν≤1，  => 0.8
        较小的ν意味着我们需要更多的弱学习器的迭代次数，通常我们用步长和迭代最大次数一起来决定算法的拟合效果
        所以这两个参数n_estimators和learning_rate要一起调参。一般来说，可以从一个小一点的ν开始调参，默认是1。
    subsample: 正则化章节讲到的子采样，取值为(0,1]
    n_iter_no_change: 分数，当验证分数没有提高时，用于决定是否使用早期停止来终止训练。
参考文档：https://zhuanlan.zhihu.com/p/58105824 第8部分
-------------'''
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300)

regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
y_1 = regr_1.predict(X)
y_2 = regr_2.predict(X)

# Plot the results
plt.figure()
plt.scatter(X, y, c="k", label="training samples")
plt.plot(X, y_1, c="g", label="n_estimators=1", linewidth=2)
plt.plot(X, y_2, c="r", label="n_estimators=300", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Boosted Decision Tree Regression")
plt.legend()
plt.show()
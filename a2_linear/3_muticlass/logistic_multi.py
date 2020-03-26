# -- coding: utf-8 --
"""
LogisticRegression 实现多分类
=============================
LogisticRegression对参数的计算采用精确解析的方式，计算时间长但模型的性能高；
SGDClassifier采用随机梯度下降/上升算法估计模型的参数，计算时间短但模型的性能较低。
"""
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.metrics import accuracy_score
import pandas as pd

data = pd.read_csv('../../data/iris.data')
# digits = datasets.load_digits()
# data = digits.data
# target = digits.target
# # 手动切分数据集
X_train = data.iloc[:-20, :-1]
y_train = data.iloc[:-20, -1:]
X_test = data.iloc[-20:, :-1]
y_test = data.iloc[-20:, -1:]
print(X_train.shape, y_train.shape)
"""
# penalty:正则化 l2/l1
# C ：正则化强度
# multi_class:多分类时使用 ovr: one vs rest
"""
lor = LogisticRegression(penalty='l1', C=100, multi_class='ovr')
"""# y_train 非one hot值"""
lor.fit(X_train, y_train)
""" 测试1 """
print(lor.score(X_test, y_test))
# """ 测试2 """
y_pred = lor.predict(X_test)
print(accuracy_score(y_test, y_pred))

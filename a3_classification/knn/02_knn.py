from __future__ import print_function
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

"""=============
# sklearn KNN
    KNN方法中没有训练过程，其分类方式就是寻找训练集附近的点。
    所以带来的一个缺陷就是计算代价非常高
    但是其思想实际上却是机器学习中普适的
# 关键参数
    n_neighbors, 邻居数，默认5
================"""

# (0) load training titanic
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

# (2) Model training

'''
# 当n_neighbors=1时出现过拟合现象
'''
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# (3) Predict & Estimate the score
# y_pred = knn.predict(X_test)
print(knn.score(X_test, y_test))

from sklearn import tree
X = [[0, 0], [2, 2], [3, 3]]    # 时间 估值
y = [0.5, 2.5, 3.1]     # 股价
"""==============
回归树
仅能预测出训练集已有的标签
================="""

clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
print(clf.predict([[10, 10]]))

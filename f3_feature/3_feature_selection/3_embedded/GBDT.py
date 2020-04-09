"""
embed GBDT方式
# GBDT作为基模型的特征选择
"""
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.datasets import load_iris

iris = load_iris()
selector = SelectFromModel(GradientBoostingClassifier()).fit(iris.data, iris.target)
print(iris.data[0:5])

# 转换数据，留下权值较大的特征
data = selector.transform(iris.data)
print(data[0:5])

# 打印权重
print(selector.estimator_.feature_importances_)
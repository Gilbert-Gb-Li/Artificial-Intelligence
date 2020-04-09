"""
# 方差选择法
先要计算各个特征的方差，然后根据阈值，选择方差大于阈值的特征。

使用feature_selection库的VarianceThreshold类来选择特征
代码如下：
"""
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import load_iris
iris = load_iris()
print(iris.data[0:5])
""" 
# 方差选择法，返回值为特征选择后的数据 
# 参数threshold为方差的阈值
"""
# fit数据
selector = VarianceThreshold(threshold=3).fit(iris.data, iris.target)
# 转换数据
data = selector.transform(iris.data)
print(data[0:5])
print(selector.variances_)

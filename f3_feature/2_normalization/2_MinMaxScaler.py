#coding: UTF-8
# 不属于同一量纲：即特征的规格不一样，不能够放在一起比较。无量纲化可以解决这一问题。
"""
# 和stanardscaler比，适合对存在极端大和小的点的数据。

# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min
"""
from sklearn.preprocessing import MinMaxScaler
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]

scaler = MinMaxScaler()
scaler.fit(data)
print(scaler.data_max_)
# scaler.transform(data)
# print(scaler.transform([[2, 2]]))

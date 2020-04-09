# coding: UTF-8
"""============
# 缺失值需要补充。
==============="""
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor

'''--------------
pandas 缺失值填充
-----------------'''
data_train = pd.read_csv("../../../data/titanic/train.csv")
print(data_train[0:10])
df = data_train.apply(lambda x: x.fillna(x.value_counts().index[0]))
print(df[0:10])

'''-------------------
# 使用随机森林计算缺失值
----------------------'''
rng = np.random.RandomState(0)
dataset = load_boston()
X_full, y_full = dataset.data, dataset.target
# X_full.shape: (506, 13)
n_samples = X_full.shape[0]
n_features = X_full.shape[1]

# estimator = RandomForestRegressor(random_state=0, n_estimators=100)
# score = cross_val_score(estimator, X_full, y_full).mean()
# print("Score with the entire dataset = %.2f" % score)

# Add missing values in 75% of the lines
missing_rate = 0.75
n_missing_samples = int(np.floor(n_samples * missing_rate))
print(n_missing_samples)
missing_samples = np.hstack((np.zeros(n_samples - n_missing_samples,
                                      dtype=np.bool),
                             np.ones(n_missing_samples,
                                    dtype=np.bool)))

print(missing_samples)
rng.shuffle(missing_samples)
print(missing_samples)
# missing_features = rng.randint(0, n_features, n_missing_samples)
# print("missing_features")
# print(missing_features)
# # Estimate the score without the lines containing missing values
# X_filtered = X_full[~missing_samples, :]
# print(X_filtered)
# y_filtered = y_full[~missing_samples]
# print(y_filtered)
# estimator = RandomForestRegressor(random_state=0, n_estimators=100)
# score = cross_val_score(estimator, X_filtered, y_filtered).mean()
# print("Score without the samples containing missing values = %.2f" % score)

# # Estimate the score after imputation of the missing values
# X_missing = X_full.copy()
# X_missing[np.where(missing_samples)[0], missing_features] = 0
# print(X_missing)
# y_missing = y_full.copy()
# print(y_missing)
# estimator = Pipeline([("imputer", Imputer(missing_values=0,
#                                           strategy="mean",
#                                           axis=0)),
#                       ("forest", RandomForestRegressor(random_state=0,
#                                                        n_estimators=100))])
# score = cross_val_score(estimator, X_missing, y_missing).mean()
# print("Score after imputation of the missing values = %.2f" % score)

'''-------------------
sklearn 空置填充
----------------------'''
from sklearn.datasets import load_iris

iris = load_iris()
data = iris.data[0:5]
print(data)
data1 = np.vstack((np.array([np.nan, np.nan, np.nan, np.nan]), data))
imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
print(data1)
imp.fit_transform(data1)

'''
# 2.
'''
from sklearn.impute import SimpleImputer
'''
# 参数：
    missing_values ：指定何种占位符表示缺失值
    strategy：插补策略，字符串，默认"mean"， "median"，"most_frequent"， "constant"，则用 fill_value 替换缺失值
'''
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit_transform([[1, 2], [np.nan, 3], [7, 6]])
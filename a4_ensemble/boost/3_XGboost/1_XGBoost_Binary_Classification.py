import pickle
import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.datasets import load_iris, load_digits, load_boston
"""
================================
Xgboost 二分类问题
1. 二分类问题
2. K折拆分数据集
================================
"""
print(__doc__)

rng = np.random.RandomState(31337)
print("Zeros and Ones from the Digits dataset: binary classification")
digits = load_digits(2)
y = digits['target']
X = digits['data']
print(X.shape, y.shape)
print(X[0], '\n', y[0])


"""
K折拆分
1. KFlod指定份数
2. kf.split(X)，数据集切分。
3. fit(X[train_index], y[train_index])，训练
4. predict(X[test_index])，预测
"""
kf = KFold(n_splits=2, shuffle=True, random_state=0)
for train_index, test_index in kf.split(X):
    xgb_model = xgb.XGBClassifier().fit(X[train_index], y[train_index])
    predictions = xgb_model.predict(X[test_index])
    actuals = y[test_index]
    print("confusion matrix")
    print(confusion_matrix(actuals, predictions))
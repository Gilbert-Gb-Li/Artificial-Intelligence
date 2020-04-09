# -*- coding: utf-8 -*-

# 改变标准输出的默认编码
import sys
import io
from sklearn import linear_model
import pandas as pd

from feature import util

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

# (1) 读取数据集
data_train = pd.read_csv("./data/train.csv")
# print(data_train.values)
# (2) 特征工程 - 处理缺失值
data_train, rfr = util.set_missing_ages(data_train)
data_train = util.set_Cabin_type(data_train)

# (3) 特特工程 - 类目型的特征离散/因子化
df = util.one_hot_encoding(data_train)

# (4) 特征工程 - 特征抽取
# 我们把需要的feature字段取出来，转成numpy格式，使用scikit-learn中的LogisticRegression建模
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.values

# y即Survival结果
y = train_np[:, 0]
# X即特征属性值
X = train_np[:, 1:]

# (5) 模型构建与训练
clf = linear_model.LogisticRegression(C=100.0, penalty='l2', tol=1e-6)

# 随机森林
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(criterion='gini', max_depth=5, n_estimators=2, verbose=0)
# 训练集精度提升明显
# clf = RandomForestClassifier(criterion='gini', max_depth=30, n_estimators=1, verbose=0)
# 可以适当减缓过拟合
# clf = RandomForestClassifier(criterion='gini', max_depth=6, n_estimators=30, verbose=0)
# xgboost
# import xgboost as xgb
# clf = xgb.XGBClassifier(max_depth=3, n_estimators=5)

# (6) 绘制learning curve
util.plot_learning_curve(clf, u"学习曲线learning curve", X, y)
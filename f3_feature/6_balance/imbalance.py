# Ref: https://elitedatascience.com/imbalanced-classes
# -*- coding: utf-8 -*-
"""============
样本均衡问题处理
==============="""
import pandas as pd
from sklearn.utils import resample

df = pd.read_csv("data/train.csv")

# print("看列名", df.columns)
# # 数据摸底
# print("看每列性质，空值和类型", df.info())
print(df[0:10])
print("看类别平衡的状况\n", df['Survived'].value_counts())

# (1) upsample / SMOTE造类似少量数据类别相似的假数据

df_majority = df[df.Survived == 0]
df_minority = df[df.Survived == 1]

# Upsample minority class
# n_samples=(len(df_majority) - len(df_minority)), 采样数量
df_minority_upsampled = resample(df_minority,
                                 replace=True,  # sample with replacement
                                 n_samples=(len(df_majority) - len(df_minority)),  # to match majority class
                                 random_state=123)  # reproducible results

# # Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled, df_minority])
print(df_upsampled.columns)
print(df_upsampled['Survived'].value_counts())

# (2) down samples
# Separate majority and minority classes
df_majority = df[df.Survived == 0]
df_minority = df[df.Survived == 1]

# down sample minority class
# n_samples=len(df_minority),    # to match majority class
df_manority_downsampled = resample(df_majority,
                                   replace=True,  # sample with replacement
                                   n_samples=len(df_minority),  # to match majority class
                                   random_state=123)  # reproducible results

# # Combine majority class with upsampled minority class
df_dwonsampled = pd.concat([df_manority_downsampled, df_minority])
print(df_dwonsampled.columns)
print(df_dwonsampled['Survived'].value_counts())

# (3) model side control
# Train model rf / svc is ok
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]
# X即特征属性值
X = train_np[:, 1:]

clf_3 = SVC(kernel='linear',
            class_weight='balanced',  # penalize
            probability=True)
# 0.638
# clf_3 = SVC(kernel='linear',
#             probability=True)
## 0.61

clf_3.fit(X, y)

# Predict on training set
pred_y_3 = clf_3.predict(X)
# How's our accuracy?
print(accuracy_score(y, pred_y_3))

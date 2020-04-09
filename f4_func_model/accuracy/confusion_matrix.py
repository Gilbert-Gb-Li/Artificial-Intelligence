from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

"""
混淆矩阵
https://blog.csdn.net/u011734144/article/details/80277225
"""

df = pd.read_csv("../../../data/titanic/train.csv")
from sklearn.metrics import accuracy_score
train_df = df.filter(regex='Survived|SibSp|Parch')
train_np = train_df.values

# y即Survival结果
y = train_np[:, 0]
# X即特征属性值
X = train_np[:, 1:]
# print(len(X[0]))

clf_3 = RandomForestClassifier()
clf_3.fit(X, y)
pred_y_3 = clf_3.predict(X)


print(accuracy_score(y, pred_y_3))
print(confusion_matrix(y, pred_y_3))
'''# 模型的分类数 #'''
labels = clf_3.classes_
'''# 将混淆矩阵转为表形式 #'''
conf_df = pd.DataFrame(confusion_matrix(y, pred_y_3), columns=labels, index=labels)
print(conf_df)

# -*- coding: utf-8 -*-
import pandas as pd
data_train = pd.read_csv("a8_titanic/data/train.csv")
print("看列名", data_train.columns)
print("看每列性质，空值和类型", data_train.info())
# print(data_train[0:1])

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
print(df[0:10])
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)


'''-----------------
filter，筛选出需要的列
顺序不一定与输入的regex字段一致
--------------------'''
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
print(train_df[0:1])
print("看列名", train_df.columns)

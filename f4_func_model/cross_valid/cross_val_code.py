from __future__ import print_function
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split, KFold, ShuffleSplit
import matplotlib.pyplot as plt

"""
更多示例，参考bagging及boost中的代码
"""

# data load
iris = load_iris()
X = iris.data
y = iris.target

# build model
# m = KNeighborsClassifier(n_neighbors=10)
m = LogisticRegression(penalty='l1', C=10, multi_class='ovr')

# cross valid
'''--------------------------
# 交叉验证
# cross_val_score，对数据集进行指定次数的交叉验证并为每次验证效果评测
# cross_val_predict，返回的是estimator的分类结果或回归值
# 交叉验证
#     estimator:估计方法对象(分类器)
#     X：数据特征(Features)
#     y：数据标签(Labels)
#     soring：调用方法(包括accuracy和mean_squared_error等等)
#     cv：几折交叉验证
#     n_jobs：同时工作的cpu个数（-1代表全部）
------------------------------'''
k_range = range(1, 10)
k_scores = []
# 多轮交叉验证
for k in k_range:
    '''
    # 由于cv=10, scores保存了10份交叉验证的结果
    # 每一份结果为在当前切分状态下的得分
    '''
    scores = cross_val_score(m, X, y, cv=10, scoring='accuracy')
    '''# 将每一轮的交叉验证结果取平均 '''
    k_scores.append(scores.mean())
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

'''----------------------------------
ShuffleSplit
将样本集合随机“打散”后再划分为训练集、测试集
-------------------------------------'''
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
scores = cross_val_score(m, X, y, cv=cv, scoring='accuracy')
print(scores)


'''-------------------
KFold, 将数据集分成K份的官方给定方案
# 切分数据集
# 切分结果返回为index
# 由于kfold参数n_splits=10，所以会打印出10次
-----------------------'''
kfold = KFold(n_splits=10, random_state=101, shuffle=True)
for train_index, test_index in kfold.split(X):
    '''
    train_index, test_index 为切分后训练集与测试集的index
    '''
    print("train index is: {}".format(train_index))
    print("test index is: {}".format(test_index))

''' 交叉验证 '''
model = RandomForestClassifier(n_estimators=10)
results = cross_val_score(model, X, y, cv=kfold)
print(results, results.mean())









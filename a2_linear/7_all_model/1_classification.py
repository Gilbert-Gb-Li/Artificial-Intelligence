# -*- coding: utf-8 -*-  
  
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB  
from sklearn.neighbors import KNeighborsClassifier   
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier  

"""
鸢尾花实例测试各种分类效果
"""
# COLOUR_FIGURE = False
seed = 215
"""
准确率计算： 正确样本/全部样本
"""
def accuracy (test_labels, pred_lables):
    correct = np.sum(test_labels == pred_lables)    
    n = len(test_labels)    
    return float(correct) / n    

def kFlod():
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
# -----------------
# 逻辑回归
# -----------------
def LR(X, y):
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    clf = LogisticRegression()
    result_set = [(clf.fit(X[train], y[train]).predict(X[test]), y) for train, test in kf.split(X)]
    score = [accuracy(y[result[1]], result[0]) for result in result_set]
    print(score)
#
# ------------------------------------------------------------------------------
# 朴素贝叶斯
# ------------------------------------------------------------------------------
# def testNaiveBayes(features, labels):
#     kf = KFold(len(features), n_folds=3, shuffle=True)
#     clf = GaussianNB()
#     result_set = [(clf.fit(features[train], labels[train]).predict(features[test]), test) for train, test in kf]
#     score = [accuracy(labels[result[1]], result[0]) for result in result_set]
#     print(score)
#
#
# # ------------------------------------------------------------------------------
# # K最近邻
# # ------------------------------------------------------------------------------
# def testKNN(features, labels):
#     kf = KFold(len(features), n_folds=3, shuffle=True)
#     clf = KNeighborsClassifier(n_neighbors=5)
#     result_set = [(clf.fit(features[train], labels[train]).predict(features[test]), test) for train, test in kf]
#     score = [accuracy(labels[result[1]], result[0]) for result in result_set]
#     print(score)
#
# #------------------------------------------------------------------------------
# #--- 支持向量机
# #------------------------------------------------------------------------------
# def testSVM(features, labels):
#     kf = KFold(len(features), n_folds=3, shuffle=True)
#     clf = svm.SVC()
#     result_set = [(clf.fit(features[train], labels[train]).predict(features[test]), test) for train, test in kf]
#     score = [accuracy(labels[result[1]], result[0]) for result in result_set]
#     print(score)
#
# #------------------------------------------------------------------------------
# #--- 决策树
# #------------------------------------------------------------------------------
# def testDecisionTree(features, labels):
#     kf = KFold(len(features), n_folds=3, shuffle=True)
#     clf = DecisionTreeClassifier()
#     result_set = [(clf.fit(features[train], labels[train]).predict(features[test]), test) for train, test in kf]
#     score = [accuracy(labels[result[1]], result[0]) for result in result_set]
#     print(score)
#
# #------------------------------------------------------------------------------
# #--- 随机森林
# #------------------------------------------------------------------------------
# def testRandomForest(features, labels):
#     kf = KFold(len(features), n_folds=3, shuffle=True)
#     clf = RandomForestClassifier()
#     result_set = [(clf.fit(features[train], labels[train]).predict(features[test]), test) for train, test in kf]
#     score = [accuracy(labels[result[1]], result[0]) for result in result_set]
#     print(score)

def scalarFeature(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)

if __name__ == '__main__': 
    data = load_iris()
    features, labels = data.data, data.target
    print(features[0])
    features = scalarFeature(features)
    print(features[0])
    print('LogisticRegression: \r')
    LR(features, labels)
    #
    # print('GaussianNB: \r')
    # testNaiveBayes(features, labels)
    #
    # print('KNN: \r')
    # testKNN(features, labels)
    #
    # print('SVM: \r')
    # testSVM(features, labels)
    #
    # print('Decision Tree: \r')
    # testDecisionTree(features, labels)
    #
    # print('Random Forest: \r')
    # testRandomForest(features, labels)
    #
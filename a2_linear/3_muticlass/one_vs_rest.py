"""
多分类OneVsRestClassifier
"""

import numpy as np
from sklearn import svm, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target

y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]      # n_classes =3

random_state = np.random.RandomState(0)
# n_samples, n_features = X.shape
# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)
'''---------------------
# 参数：
    estimator, 基本分类器
    n_jobs, 并行度
-------------------------'''
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state),
                                 n_jobs=-1)
# classifier = OneVsRestClassifier(LogisticRegression(C=1e-3),
#                                  n_jobs=-1)

classifier.fit(X_train, y_train)

'''返回距每个类边界的距离'''
y_distance = classifier.decision_function(X_test)
y_pred = classifier.predict(X_test)
y_proba = classifier.predict_proba(X_test)

# print('pred: \n', y_pred[0])
# print('proba:\n', y_proba[0])
# print('distance: \n', y_distance[0])


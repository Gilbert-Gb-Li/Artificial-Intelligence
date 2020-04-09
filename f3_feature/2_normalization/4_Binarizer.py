#coding: UTF-8
"""
# 二值化
# 信息冗余：对于某些定量特征，其包含的有效信息为区间划分，
#   例如学习成绩，假若只关心“及格”或不“及格”，那么需要将定量的考分，转换成“1”和“0”表示及格和未及格。
"""
from sklearn.preprocessing import Binarizer

X = [[1., -1.,  2.], [2.,  0.,  0.], [0.,  1., -1.]]
"""# fit does nothing"""
binarizer = Binarizer().fit(X)
y = binarizer.transform(X)
print(y)
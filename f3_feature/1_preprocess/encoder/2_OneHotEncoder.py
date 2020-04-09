# coding: UTF-8
"""
# sklearn 对定性特征哑编码可以达到非线性的效果。
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder

'''
参数：
    categorical_features = 'all'，指定了对哪些特征进行编码
    sparse=True，稀疏的格式，指定 False 则就不用 toarray()
    handle_unknown=’error’，如果碰到未知的类别，是返回一个错误还是忽略它
'''

enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
"""
# feature 为二维数组，
# 每行为一个样本
# 每列为一个特征
"""
featrue = np.array([["男", "X"],
                    ["女", "Y"],
                    ["男", "Z"],
                    ["男", "Y"]
          ])
one_hot = enc.fit_transform(featrue)
print(one_hot)

''' one hot 还原'''
label = enc.inverse_transform(one_hot)
print(label)
# print(enc.transform([["女", "Z"], ["男", "Y"]]).toarray())


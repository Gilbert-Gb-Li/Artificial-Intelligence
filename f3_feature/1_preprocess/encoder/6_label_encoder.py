# coding: UTF-8
"""
# 对定性特征将n个类别编码为0~n-1之间的整数。
"""
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

import numpy as np

sex = np.array(["male", "female", "female", "male"])
le = LabelEncoder()
le.fit(sex)
sex = le.transform(sex)
# sex.shape: (1, 4)
print(sex)


lb = LabelBinarizer()
y = lb.fit_transform(sex)
# y.shape: (4, 1)
print(y)
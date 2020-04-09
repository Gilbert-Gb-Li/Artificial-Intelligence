import numpy as np
import pandas as pd
"""
数值分桶
"""

n_samples = 10
'''# 设置随机种子 #'''
np.random.seed(1)
a = np.random.randint(0, 10, n_samples)
boundaries = [0, 4, 9]
'''-------------------
# pandas 分段函数
# 参数：
    bins  : 切割后的区间
    right : bins区间设定, 超出范围为Nan
            True, (0,4,9]   False, [0,4,9)
    labels: 分割后的bins打标签，须比bins少1
        返回每个元素对应的类别
----------------------'''
a_discretized_1 = pd.cut(a, bins=boundaries, right=False, labels=['a', 'b'])
a_discretized_2 = pd.cut(a, bins=boundaries, labels=range(len(boundaries) - 1), right=False)
a_discretized_3 = pd.cut(a, bins=boundaries, labels=range(len(boundaries) - 1), right=False).astype(float)
print(a, '\n')
print(a_discretized_1, '\n', a_discretized_1.dtype, '\n')
print(a_discretized_2, '\n', a_discretized_2.dtype, '\n')
print(a_discretized_3, '\n', a_discretized_3.dtype, '\n')

# https://stackoverflow.com/questions/23267767/how-to-do-discretization-of-continuous-attributes-in-sklearn
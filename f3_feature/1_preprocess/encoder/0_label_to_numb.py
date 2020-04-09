"""
手工数值化
"""
from sklearn.datasets import load_iris
import numpy as np
iris = load_iris()
label = np.array(["m", "f", "m", "m", "f", "f"])
name = set(label)
d = {itr: idx for idx, itr in enumerate(name)}
print(d)
n_label = [d[itr] for itr in label]
print("原始字符数据：", label)
print("转换后数据：", n_label)
"""
高斯朴素贝叶斯模拟EM算法
"""

import matplotlib.pyplot as plt
from scipy.linalg import norm
import numpy.matlib as ml
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.naive_bayes import GaussianNB
np.random.seed(0)

X, _ = make_blobs(n_samples=1500, random_state=170)
y = np.random.randint(0, 3, len(X)) 
gnb = GaussianNB()
for step in range(10):
    # M步
    gnb.fit(X, y) 
    # E步
    y = gnb.predict(X) 
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
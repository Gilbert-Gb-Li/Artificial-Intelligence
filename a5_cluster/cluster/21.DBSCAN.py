from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
"""
层次聚类
可完成复杂数据聚类
"""
centers = [[0, 1], [-1, -1], [1, -1]]
X, y = make_blobs(n_samples=1500, random_state=170)
trs = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X = np.dot(X, trs)

"""
DBSCAN 聚类
============
参数：
    eps，半径
    min_samples，核心点的判定
属性：
    metric ：度量方式，默认为欧式距离，还有metric=’precomputed’（稀疏半径邻域图）
    core_sample_indices_ : 核心点的索引，因为labels_不能区分核心点还是边界点，所以需要用这个索引确定核心点
    components_：训练样本的核心点
    labels_：每个点所属集群的标签，-1代表噪声点    
"""
clt = DBSCAN(eps=0.5, min_samples=30)
yp = clt.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=yp, edgecolors='k')
plt.axis("equal")
plt.show()
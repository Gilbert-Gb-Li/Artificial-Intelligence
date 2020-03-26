from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.cluster import MeanShift

from func_model.visual._20_plot_ge_2d import contourf

centers = [[1, 1], [-1, -1], [1, -1]]
X, y = make_blobs(n_samples=1000, centers=centers, cluster_std=0.6, random_state=1)
clt = MeanShift(bandwidth=1)
clt.fit(X)

contourf(clt, X, y)

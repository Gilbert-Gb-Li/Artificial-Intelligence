import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
"""
PCA 降维
"""
print(__doc__)

iris = datasets.load_iris()
# X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

'''使用PCA降维'''
X_reduced = PCA(n_components=3).fit_transform(iris.data)


fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
plt.show()
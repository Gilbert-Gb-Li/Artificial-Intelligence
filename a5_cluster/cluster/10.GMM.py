import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn import mixture
from sklearn.datasets import make_blobs


n_samples = 300
"""
高斯混合模型
==============
参数：
    n_components:混合高斯模型个数，默认为1
    tol：EM迭代停止阈值，默认为1e-3.
    max_iter:最大迭代次数，默认100
    n_init:初始化次数，用于产生最佳初始参数，默认为1 
    random_state :随机数发生器
    warm_start :若为True，则fit时会以上一次fit结果作为初始化参数，
        适合相同问题多次fit的情况，能加速收敛，默认为False。
"""

centers = [[0, 1], [-1, -1], [1, -1]]

X, y = make_blobs(n_samples=1500, random_state=170)
trs = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X = np.dot(X, trs)

clf = mixture.GaussianMixture(max_iter=300, n_components=3)
clf.fit(X)

# 绘图
x_min, x_max = X[:, 0].min() - .2, X[:, 0].max() + .2
y_min, y_max = X[:, 1].min() - .2, X[:, 1].max() + .2
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# 返回加权对数概率，所以指数形式，就是gmm模型给出的概率
Z = clf.score_samples(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

YY = clf.predict(X)
from mpl_toolkits.mplot3d import Axes3D
print(YY[:10])
mpl.style.use('fivethirtyeight')
# 绘制3d图
fig = plt.figure()
ax = fig.gca(projection='3d')
CS = ax.plot_surface(xx, yy, Z, alpha=0.2)
ax.scatter(X[:, 0], X[:, 1], .8)

plt.title('GMM')
plt.axis('tight')
plt.show()


# # generate random sample, two components
# np.random.seed(0)
#
# # generate spherical data centered on (20, 20)
# shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])
#
# # generate zero centered stretched Gaussian data
# C = np.array([[0., -0.7], [3.5, .7]])
# stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)
#
# # concatenate the two datasets into the final training set
# X_train = np.vstack([shifted_gaussian, stretched_gaussian])
#
# # fit a Gaussian Mixture Model with two components
#
# clf = mixture.GaussianMixture(max_iter=300, n_components=2, covariance_type='full')
# clf.fit(X_train)
#
# # display predicted scores by the model as a contour plot
# x = np.linspace(-20., 30.)
# y = np.linspace(-20., 40.)
# X, Y = np.meshgrid(x, y)
# XX = np.array([X.ravel(), Y.ravel()]).T
#
# Z = clf.score_samples(XX)
# Z = Z.reshape(X.shape)
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification

# from func_model.visual.v_simple import contourf

X, y = make_classification(n_features=3, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1, class_sep=0.2)

# X, y = make_circles(noise=0.1, factor=0.1, random_state=1)
# X, y = make_blobs(n_samples=1500, random_state=170)
# print(X.shape)

"""
Kmeans 算法应用
========================
参数：
    1) n_clusters: 即k值，一般需要多试一些值以获得较好的聚类效果。
    2) max_iter： 最大的迭代次数，如果数据集不是凸的，可能很难收敛，指定最大的迭代次数让算法可以及时退出循环。 
    3) n_init：用不同的初始化质心运行算法的次数。
        由于K-Means是结果受初始值影响的局部最优的迭代算法，因此需要多跑几次以选择一个较好的聚类效果，默认是10，一般不需要改。
        如果你的k值较大，则可以适当增大这个值。 
    4) init： 即初始值选择的方式，可以为完全随机选择’random’,优化过的’k-means++’或者自己指定初始化的k个质心。一般建议使用默认的’k-means++’。 
    5) algorithm：有“auto”, “full” or “elkan”三种选择。
        ”full”就是我们传统的K-Means算法， “elkan”是（机器学习(25)之K-Means聚类算法详解）原理篇讲的elkan K-Means算法。
        默认的”auto”则会根据数据值是否是稀疏的，来决定如何选择”full”和“elkan”。
        一般数据是稠密的，那么就是 “elkan”，否则就是”full”。一般来说建议直接用默认的”auto”
"""
clt = KMeans(n_clusters=3, random_state=0)
print(clt.fit_transform(X))
print(clt.fit_predict(X))

print(clt.cluster_centers_)
# print(clt.predict(X))

# contourf(clt, X, y, proba=False)

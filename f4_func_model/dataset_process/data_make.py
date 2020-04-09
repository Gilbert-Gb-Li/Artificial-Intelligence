import numpy as np


def data_gen():
    """
    # 生成数据
    X, y = make_classification(n_samples=10000,        # 样本个数
                               n_features=25,          # 特征个数
                               n_informative=3,        # 有效特征个数
                               n_redundant=2,          # 冗余特征个数（有效特征的随机组合）
                               n_repeated=0,           # 重复特征个数（有效特征和冗余特征的随机组合）
                               n_classes=3,            # 样本类别
                               n_clusters_per_class=1, # 簇的个数
                               random_state=0)
    返回值：
        X：形状数组[n_samples，n_features], 生成的样本。
        y：形状数组[n_samples], 每个样本的类成员的整数标签。
    """
    from sklearn.datasets import make_classification, make_moons
    '''
    # 生成环形数据
    # factore表示有0.1的点是异常点 ：外圈与内圈的尺度因子<1
    # 里圈代表一个类，外圈也代表一个类.noise表示有0.1的点是异常点
    '''
    # X, y = make_circles(noise=0.1, factor=0.1, random_state=1)

    '''# 生成半环形图'''
    # X, y = make_moons(n_samples=300, noise=.05)

    '''
    # make_blobs函数是为聚类产生数据集
    # 参数：
    #     n_features, 每个样本的特征数
    #     centers, 要生成的样本中心（类别）数，或者是确定的中心点
    #     cluster_std, 每个类别的方差，例:若生成2类数据方差不同，可以将cluster_std设置为[1.0,3.0]
    #     center_box：中心确定之后的数据边界
    # 返回：
    #     X, 生成的样本数据集
    #     y, 样本数据集的标签
    '''
    # X, y = make_blobs(n_samples=1500, random_state=170)

    X, y = make_classification(n_samples=500, n_features=25, random_state=100)
    # print(type(X), X.shape, y.shape)
    # print(X[:1, :])
    # print(y[:3])
    return X, y


def multi_label():
    """
    sparse（稀疏）:如果是True，返回一个稀疏矩阵，稀疏矩阵表示一个有大量零元素的矩阵。
    n_labels:每个实例的标签的平均数量。
    return_indicator:“sparse”在稀疏的二进制指示器格式中返回Y。
    allow_unlabeled:如果是True，有些实例可能不属于任何类。
    """
    from sklearn.datasets import make_multilabel_classification
    X, y = make_multilabel_classification(sparse=True, n_labels=5,
                                          return_indicator='sparse', allow_unlabeled=False)
    return X, y


"""
2.
生成多元正态分布数据
参数：
    mean    ： mean是多维分布的均值维度为1；
    cov     ： 协方差矩阵，注意：协方差矩阵必须是对称的且需为半正定矩阵；
    size    ： 指定生成的正态分布矩阵的维度,  例： (2, 2), 100 # (100,)
               若size=(1, 1, 2)，则输出的矩阵的shape即形状为 1,1,2,N（N为mean的长度）。
    check_valid：这个参数用于决定当cov即协方差矩阵不是半正定矩阵时程序的处理方式： warn，raise以及ignore。
        warn，输出警告但仍旧会得到结果；
        raise，报错且不会计算出结果；
        ignore，忽略并计算出结果。
"""


def multi_normal():
    mean = (1, 1)
    cov = [[1, 0], [0, 1]]
    size = (1, 1)
    # x (1,1,2): [[[1.28748292 0.22326428]]]
    # ---------------- #
    # mean = (1, 1, 0)
    # cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # size = (1, 1)
    # x: (1, 1, 3): [[[ 1.78867662  3.26287102 -0.15716529]]]
    x = np.random.multivariate_normal(mean, cov, size)
    print(x.shape)

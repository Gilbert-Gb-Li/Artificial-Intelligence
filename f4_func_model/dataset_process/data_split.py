from __future__ import print_function
from sklearn.model_selection import train_test_split

"""=============
数据集切分
================"""


def dataset_split(X, y):
    """
    2. 可以调用sklearn里的数据切分函数：
    参数：
        train_data：被划分的样本特征集
        train_target：被划分的样本标签
        test_size：如果是浮点数，在0-1之间，表示样本占比；如果是整数的话就是样本的数量
        random_state：是随机数的种子。
    返回numpy数组
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.3, random_state=4)
    return X_train, X_test, y_train, y_test


def manual_split(X, y):
    n_samples = len(X)
    X_train = X[:int(.9 * n_samples)]
    y_train = y[:int(.9 * n_samples)]
    X_test = X[int(.9 * n_samples):]
    y_test = y[int(.9 * n_samples):]
    return X_train, y_train, X_test, y_test



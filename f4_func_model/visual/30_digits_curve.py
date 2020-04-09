"""
学习曲线
-------------
    # 1.现将所有样本用交叉验证方法或者（随机抽样方法) 得到 K对 训练集-验证集
    # 2.依次对K个训练集，拿出数量不断增加的子集如m个，并在这些K*m个子集上训练模型。
    # 3.依次在对应训练集子集、验证集上计算得分。
    # 4.对每种大小下的子集，计算K次训练集得分均值和K次验证集得分均值，共得到m对值。
    # 5.绘制学习率曲线。x轴训练集样本量，y轴模型得分或预测准确率。
"""

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn import linear_model


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    :param train_sizes:
        np.linspace(.1, 1.0, 5)=array([ 0.1  ,  0.325,  0.55 ,  0.775,  1.   ])，
        表示训练样本从总数据集中分别取出10%、32.5%、55%、77.5%、100%作为子数据集。
        对当前子数据集，根据cv定好的规则划分为训练集和测试集，然后使用estimator指定的模型计算测试集得分。
        现在是回归问题，因此函数learning_curve 的score是MSE
    # 参数
        # estimator : 你用的分类器。
        # title : 表格的标题。
        # X : 输入的feature，numpy类型
        # y : 输入的target vector
        # title：图像的名字
        # cv：cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
        #     默认cv=None，如果需要传入则如下：
        #     cv:int,交叉验证生成器或可迭代的可选项，确定交叉验证拆分策略。
        #     cv的可能输入是：
        #        - 无，使用默认的3倍交叉验证，
        #        - 整数，指定折叠数。
        #        - 要用作交叉验证生成器的对象。
        #        - 可迭代的yielding训练/测试分裂。
        # ShuffleSplit：我们这里设置cv，交叉验证使用ShuffleSplit方法，
        #     一共取得100组训练集与测试集，每次的测试集为20%，
        #     它返回的是每组训练集与测试集的下标索引，由此可以知道哪些是train，那些是test。
        # ylim：tuple, shape (ymin, ymax), 可选的。定义绘制的最小和最大y值，这里是（0.7，1.01）。
        # n_jobs : 整数，可选并行运行的作业数（默认值为1）
        # train_sizes：指定训练样本数量的变化规则，
        #    比如np.linspace(.1, 1.0, 5)表示把训练样本数量从0.1~1分成五等分，
        #    从[0.1, 0.33, 0.55, 0.78, 1. ]的序列中取出训练样本数量百分比，
        #    逐个计算在当前训练样本数量情况下训练出来的模型准确性。
    :return: 绘图实例
    """
    plt.figure()
    plt.title(title)

    '''---------------
    # ylim定义绘制的最小和最大y值
    ------------------'''
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    '''---------------------------
    # train_scores.shape = (5, 10)，test_scores.shape = (5, 10)，共有5个子数据集。
    # 每个子数据集根据cv随机划分出10组不同的（训练集 + 测试集）
    --------------------------------'''
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


if __name__ == '__main__':
    digits = load_digits()
    X, y = digits.data, digits.target

    title = "Learning Curves (LR)"

    '''-------------------------
    实现交叉验证、或 随机抽样划分不同的训练集合验证集
    # 随机选择20%的数据集作为测试集，80%作为训练集。然后重复10次，共选出10份(训练数据+测试数据)
    ----------------------------'''
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    estimator = linear_model.LogisticRegression()

    plt1 = plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv)
    plt1.show()

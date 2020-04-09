"""
等值线图绘制
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.style.use('fivethirtyeight')


def contourf(model, X, y, proba=True):
    # 调整图片风格
    # 定义xy网格，用于绘制等值线图
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    ''' 
    # x以及y的数据，从指定的min到max值，以0.1为步长，生成一维向量
    # 组成矩阵
    # x1.shape              : (27,)
    # y1.shape              : (30,)
    # xx.shape == yy.shape  : (30, 27)
    # x1 == xx[0,1,2...], 横坐标
    # y1 == yy[0,1,2...], 纵坐标
    '''
    x1 = np.arange(x_min, x_max, 0.1)
    y1 = np.arange(y_min, y_max, 0.1)
    xx, yy = np.meshgrid(x1, y1)

    '''
    # 预测可能性
    # predict_proba 返回的是一个 n 行 k 列的数组，
    # 第 i 行 第 j 列上的数值是模型预测 第 i 个预测样本为某个标签的概率，并且每一行的概率和为1。
    '''
    # xx.ravel().shape: 810, 30*27
    # np.c_[xx.ravel(), yy.ravel()].shape: (810, 2)
    if proba:
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=.8)
    '''# 绘制散点图'''
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title("figure")
    plt.axis("equal")

    plt.show()


def v_pcolormesh(model, X, y, proba=True):
    """
    作用在于能够直观表现出分类边界， 分类区域
    """
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    x1 = np.arange(x_min, x_max, 0.1)
    y1 = np.arange(y_min, y_max, 0.1)
    xx, yy = np.meshgrid(x1, y1)
    if proba:
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    '''
    plt.pcolormesh()会根据y_predict的结果自动在cmap里选择颜色
    '''
    plt.pcolormesh(xx, yy, Z, alpha=.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title("figure")
    plt.axis("equal")
    plt.show()


def v_figure():
    """
    num     : 图像编号或名称
    figsize : 指定figure的宽和高，单位为英寸
    dpi     : 每英寸多少个像素，缺省值为80
    facecolor: 背景颜色
    edgecolor: 边框颜色
    frameon  : 是否显示边框
    """
    plt.figure(1, figsize=(4, 3))


def plot_3d():
    """
    绘制3d图
    -- 折线图
    """
    # 必须导入，否则报错
    from mpl_toolkits.mplot3d import Axes3D
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z ** 2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    ax.plot(x, y, z)
    ax.legend()
    plt.show()

if __name__ == '__main__':
    plot_3d()
#by cangye@hotmail.com
#引入库
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
"""
手写识别
使用单层神经网络
手工实现
=======================
"""

"""1.获取数据"""
mnist = input_data.read_data_sets("data/", one_hot=True)
"""1.1 数据查看"""
# X, y = mnist.train.next_batch(100)
# X.shape: (100, 784)
# y.shape: (100, 10)
# y[0]: [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.] one-hot
# plt.matshow(X[0].reshape(28,28))

"""2.构建网络模型"""
"""2.1 x，label分别为图形数据和标签数据"""
x = tf.placeholder(tf.float32, [None, 784])
label = tf.placeholder(tf.float32, [None, 10])

"""
2.2 单层网络参数，权值和偏置
    初值不要用全零，影响收敛
"""
W = tf.Variable(tf.random_normal([784, 10]))
"""
# get_variable随机初始化
# W = tf.get_variable('W', [784, 10])
"""
b = tf.Variable(tf.zeros([10]))

"""2.3 目标函数，本例中sigmoid激活函数"""
y = tf.nn.sigmoid(tf.matmul(x, W) + b)

"""
2.4 定义损失函数为欧氏距离
# axis缺省值为none，表示对所有元素求平均
# axis=0 按列计算 
# axis=1 按行计算 
"""
loss = tf.reduce_mean(tf.square(y-label))

"""2.5 用梯度迭代算法"""
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
"""
# 2.6 定义验证方法
# tf.argmax()返回最大数值的下标
# axis=0 按列找 
# axis=1 按行找 
# tf.argmax(y,axis=1)，获取矩阵y中列的最大值的索引
# tf.equal返回boolean
# tf.cast类型转换
"""
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""3. 定义会话"""
sess = tf.Session()
# 初始化所有变量
sess.run(tf.global_variables_initializer())
"""迭代过程"""
# 将当前默认graph写入文件
# train_writer = tf_FC.summary.FileWriter("mnist-logdir", sess.graph)
for itr in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    res = sess.run([train_step, accuracy], feed_dict={x: batch_xs, label: batch_ys})
    if itr % 100 == 0:
        print("step:%6d  accuracy:" % itr, sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        label: mnist.test.labels}))


# ################################绘图过程################################################
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.style.use('fivethirtyeight')
'''# 获取W取值'''
W = sess.run(W.value())
# print(W)
# 绘图过程
fig = plt.figure()
ax = fig.add_subplot(221)
ax.matshow(np.reshape(W[:, 1], [28, 28]), cmap=plt.get_cmap("Purples"))
ax = fig.add_subplot(222)
ax.matshow(np.reshape(W[:, 2], [28, 28]), cmap=plt.get_cmap("Purples"))
ax = fig.add_subplot(223)
ax.matshow(np.reshape(W[:, 3], [28, 28]), cmap=plt.get_cmap("Purples"))
ax = fig.add_subplot(224)
ax.matshow(np.reshape(W[:, 4], [28, 28]), cmap=plt.get_cmap("Purples"))
plt.show()
# by cangye@hotmail.com
# TensorFlow入门实例
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


# 定义summary函数
def variable_summaries(var, name="layer"):
    with tf.variable_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# 读取数据
data = pd.read_csv("data/iris.data.csv")
# 获取种类
c_name = set(data.name.values)
print(c_name)
# 数据预处理
iris_label = np.zeros([len(data.name.values), len(c_name)])
iris_data = data.values[:, :-1]
iris_data = iris_data - np.mean(iris_data, axis=0)
iris_data = iris_data / np.max(iris_data, axis=0)
# 划分训练集合测试集，这里二者不重合，为了观测overfiting，但是他么数据太简单没有过拟合！！
train_data = []
train_data_label = []
test_data = []
test_data_label = []
for idx, itr_name in enumerate(c_name):
    datas_t = iris_data[data.name.values == itr_name, :]
    labels_t = np.zeros([len(datas_t), len(c_name)])
    labels_t[:, idx] = 1
    train_data.append(datas_t[:30])
    train_data_label.append(labels_t[:30])
    test_data.append(datas_t[30:])
    test_data_label.append(labels_t[30:])
train_data = np.concatenate(train_data)
train_data_label = np.concatenate(train_data_label)
test_data = np.concatenate(test_data)
test_data_label = np.concatenate(test_data_label)
# 定义feed接收部分
x = tf.placeholder(tf.float32, [None, 4], name="input_x")
label = tf.placeholder(tf.float32, [None, 3], name="input_y")
'''-------------------------------------------
# 多层神经网络效果反而不好，梯度消失，出现过拟合现象
    1. 使用relu来激活函数替代
    2. 使用batch_normal
----------------------------------------------'''
net = tf.layers.dense(x, 28, activation=tf.nn.relu)
# net = tf_FC.layers.dense(x, 28, activation=tf_FC.nn.sigmoid)
# net = tf_FC.layers.batch_normalization(net, training=True)
net = tf.layers.dense(net, 28, activation=tf.nn.relu)
# net = tf_FC.layers.dense(net, 28, activation=tf_FC.nn.sigmoid)
# net = tf_FC.layers.batch_normalization(net, training=True)
net = tf.layers.dense(net, 28, activation=tf.nn.relu)
# net = tf_FC.layers.dense(net, 28, activation=tf_FC.nn.sigmoid)
# net = tf_FC.layers.batch_normalization(net, training=True)
net = tf.layers.dense(net, 28, activation=tf.nn.relu)
# net = tf_FC.layers.dense(net, 28, activation=tf_FC.nn.sigmoid)
# net = tf_FC.layers.batch_normalization(net, training=True)
net = tf.layers.dense(net, 28, activation=tf.nn.relu)
# net = tf_FC.layers.dense(net, 28, activation=tf_FC.nn.sigmoid)
# net = tf_FC.layers.batch_normalization(net, training=True)
net = tf.layers.dense(net, 28, activation=tf.nn.relu)
# net = tf_FC.layers.dense(net, 28, activation=tf_FC.nn.sigmoid)
# net = tf_FC.layers.batch_normalization(net, training=True)
y = tf.layers.dense(net, 3, activation=tf.nn.sigmoid)
# net = tf_FC.layers.batch_normalization(net, training=True)
'''# 定义损失函数(这里应该用交叉熵，需要自行修改) #'''
loss = tf.reduce_mean(tf.square(y - label))
'''# 定义精确度 #'''
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# train_step = tf_FC.train.GradientDescentOptimizer(0.5).minimize(loss)
# 定义求解器
optimizer = tf.train.GradientDescentOptimizer(0.5)
'''-------------------------------------
没看懂？？？
# 这个部分是最重要的
# tf_FC.trainable_variables()返回一个神经网络可训练的权值列表
# 但是我们又进行了筛选，名字中有w的作为一类，其他作为一类
----------------------------------------'''
var_list_w = [var for var in tf.trainable_variables() if "kernel" in var.name]
var_list_b = [var for var in tf.trainable_variables() if "bias" in var.name]
# 这里来计算相关的梯度
# 每个分量单独计算梯度
gradient_w = optimizer.compute_gradients(loss, var_list=var_list_w)
gradient_b = optimizer.compute_gradients(loss, var_list=var_list_b)
# 将梯度内容加入summary
for idx, itr_g in enumerate(gradient_w):
    variable_summaries(itr_g[0], "layer%d-w-grad" % idx)
for idx, itr_g in enumerate(gradient_b):
    variable_summaries(itr_g[0], "layer%d-b-grad" % idx)
for idx, itr_g in enumerate(var_list_w):
    variable_summaries(itr_g, "layer%d-w" % idx)
for idx, itr_g in enumerate(var_list_b):
    variable_summaries(itr_g, "layer%d-b" % idx)
# 将执行梯度下降步骤
train_step = optimizer.apply_gradients(gradient_w + gradient_b)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 1.
train_writer = tf.summary.FileWriter("logdir", sess.graph)
merged = tf.summary.merge_all()
# 这里就是迭代过程了
for itr in range(600):
    sess.run(train_step, feed_dict={x: train_data, label: train_data_label})
    if itr % 30 == 0:
        acc1 = sess.run(accuracy, feed_dict={x: train_data,
                                             label: train_data_label})
        acc2 = sess.run(accuracy, feed_dict={x: test_data,
                                             label: test_data_label})
        print("step:{:6d}  train:{:.3f} test:{:.3f}".format(itr, acc1, acc2))
        summary = sess.run(merged,
                           feed_dict={x: train_data,
                                      label: train_data_label})
        # 2.
        train_writer.add_summary(summary, itr)
data1, data2 = sess.run(gradient_b[0], feed_dict={x: train_data,
                                                  label: train_data_label})
print(np.shape(data1), np.shape(data2))

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use('fivethirtyeight')
xx = yy = np.arange(-1.0, 1.0, 0.05)
X, Y = np.meshgrid(xx, yy)
sp = np.shape(X)
xl = np.reshape(X, [-1, 1])
yl = np.reshape(Y, [-1, 1])
gridxy = np.concatenate([xl, yl], axis=1)
print(np.shape(gridxy))
zl = sess.run(y, feed_dict={x: gridxy})
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.axis('equal')
print(len(iris_data))
surface = ax.plot_surface(X, Y, np.reshape(zl[:, 0], sp), alpha=0.2)
surface = ax.plot_surface(X, Y, np.reshape(zl[:, 1], sp), alpha=0.2)
surface = ax.plot_surface(X, Y, np.reshape(zl[:, 2], sp), alpha=0.2)
ax.scatter(iris_data[:50, 0], iris_data[:50, 1], color="#990000", s=60)
ax.scatter(iris_data[50:100, 0], iris_data[50:100, 1], color="#009900", s=60)
ax.scatter(iris_data[100:, 0], iris_data[100:, 1], color="#000099", s=60)
plt.show()

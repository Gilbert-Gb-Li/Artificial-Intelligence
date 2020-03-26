# by cangye@hotmail.com

"""
贷款欺诈高阶api实现
"""

import pandas as pd
import tensorflow as tf
import numpy as np
data = pd.read_csv("../data/creditcard.csv")
# data.columns:
# ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
# 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
# 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
# 'Class']

# 1. 切分数据集
class1 = data[data.Class == 0]
class2 = data[data.Class == 1]
print(len(class1))
print(len(class2))
print(np.shape(class1.values))
# 2.
'''values 去表头索引将DataFrame转换为ndarray'''
data1 = class1.values
data2 = class2.values
x = tf.placeholder(tf.float32, [None, 28], name="input_x")
label = tf.placeholder(tf.float32, [None, 2], name="input_y")

"""
3. 使用高阶api
# 对于sigmoid激活函数而言，效果可能并不理想
# tf.layers.dense: outputs = activation(inputs * kernel + bias)
# 参数:
#   x, 输入
#   units, 输出唯独
#   activation, 激活函数
"""
net = tf.layers.dense(x, 28, activation=tf.nn.relu)
net = tf.layers.dense(net, 28, activation=tf.nn.relu)
y = tf.layers.dense(net, 2, activation=tf.nn.sigmoid)
# 4. loss
loss = tf.reduce_mean(tf.square(y-label))
# 正则化
# loss_w = [tf.nn.l2_loss(var) for var in tf.trainable_variables() if "kernel" in var.name]
# weights_norm = tf.reduce_sum(loss_w)
# loss = tf.reduce_mean(tf.square(y-label))+0.01*weights_norm

# 5. accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 6. derivative
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 7. iterate
for itr in range(3000):
    idx1 = np.random.randint(284000)
    idx2 = np.random.randint(400)
    """
    正负样本各挑选25个训练
    # 去除了时间列和标签列
    """
    feedx = np.concatenate([data1[idx1: idx1+25, 1: 29],
                            data2[idx2: idx2+25, 1: 29]])
    """
    创建50个样本，并one-hot编码
    前25个赋值为1
    后25个赋值为0
    [1. 0.]] [[0. 1.]
    """
    feedy = np.zeros([50, 2])
    feedy[:25, 0] = 1
    feedy[25:, 1] = 1
    sess.run(train_step, feed_dict={x: feedx, label: feedy})
    if itr % 30 == 0:
        feedx = np.concatenate([data1[3000:3000+400, 1:29],
                                data2[:400, 1:29]])
        feedy = np.zeros([800, 2])
        feedy[:400, 0] = 1
        feedy[400:, 1] = 1
        print("step:%6d  accuracy:"% itr, 100*sess.run(accuracy, feed_dict={x: feedx,
                                        label: feedy}))


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.style.use('fivethirtyeight')
xx = yy = np.arange(-20.0, 20.0, 0.1)
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
surface = ax.plot_surface(X, Y, np.reshape(zl[:, 0], sp), alpha=1)
surface = ax.plot_surface(X, Y, np.reshape(zl[:, 1], sp), alpha=1)
ax.scatter(data1[3000:3000+400, 1], data1[3000:3000+400, 2], color="#990000", s=60)
ax.scatter(data2[:400, 1], data2[:400, 2], color="#009900", s=60)
plt.show()


mpl.style.use('fivethirtyeight')
data_in = np.concatenate([data1[3000:3000+400, 1:4],
                                data2[:400, 1:4]])
zl = sess.run(tf.argmax(y, 1), feed_dict={x: data_in})
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.axis('equal')
print(zl)
ax.scatter(data_in[zl==0, 0], data_in[zl==0, 1], data_in[zl==0, 2], color="#990000", s=60, alpha=0.1)
ax.scatter(data_in[zl==1, 0], data_in[zl==1, 1], data_in[zl==1, 2], color="#000099", s=60, alpha=0.1)
plt.show()
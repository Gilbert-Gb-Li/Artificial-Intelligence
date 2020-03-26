import tensorflow as tf

"""
线性回归tf手工实现
每步计算X,Y的维度参见实例[mnist/single-layer-crossentropy.py]
================
"""

"""
1. 定义需要从外界接收样本：placeholder 
    # 只有一列所以为[none, 1]
    # None表示行数不限
"""
x = tf.placeholder(tf.float32, [None, 1])
d = tf.placeholder(tf.float32, [None, 1])
"""
2. 需要定义可训练参数：Variable
"""
w = tf.get_variable("w", [1, 1])
b = tf.get_variable("b", [1])

"""
3. 需要定义目标函数
模型: y=x * w + b
返回： 
    若x为[m * n] 矩阵，则
    y.shape: (m, n)
"""
y = tf.matmul(x, w) + b

"""
# 4.定义损失函数：loss函数 
# 4.1 得到关于每个预测值的是矩阵
loss.shape: (m, n)
"""
loss = (y - d) ** 2
"""4.2 得出loss均值[reduce_mean] """
loss = tf.reduce_mean(loss)

"""
# 定义优化器，用于为梯度计算及迭代
# 使用该优化器自动求导
# 参数 0.1 为学习率
"""
optimizer = tf.train.GradientDescentOptimizer(0.1)
grads = optimizer.compute_gradients(
    loss,
    [w, b])

# 执行：w=w-eta*dw
train_step = optimizer.apply_gradients(grads)

# 输入样本进行训练
# 定义会话 
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 读取训练数据
import numpy as np

file = np.load("homework.npz")
data_x = file['X']
data_d = file['d']

"""
5. 实际执行
需显式指定迭代
"""
for step in range(200):
    """
    # 主要训练train_step，即每次参数的变化 w=w-eta*dw； 只运行最后一步即可 
    # 同时也可以添加其他需要的计算值
    """
    st, ls = sess.run(
        [train_step, loss],
        feed_dict={
            x: data_x,
            d: data_d
        }
    )
    print(ls)

"""6 输出预测值"""
pred_y = sess.run(y, feed_dict={x: data_x})

import matplotlib.pyplot as plt

plt.scatter(data_x[:, 0], data_d[:, 0], c="y")
plt.scatter(data_x[:, 0], pred_y[:, 0], c="r")
plt.show()

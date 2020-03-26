import tensorflow as tf  
"""
线性回归高阶API实现
=================
"""

"""# 需要从外界接收样本：placeholder"""
x = tf.placeholder(tf.float32, [None, 1]) 
d = tf.placeholder(tf.float32, [None, 1])

"""
# 需要定义可训练参数：Variable 
# 多层网络形式，等同于 tf_FC.layers.dense形式
w1 = tf.get_variable("w1", [1, 1000]) 
b1 = tf.get_variable("b1", [1000]) 
w2 = tf.get_variable("w2", [1000, 1]) 
b2 = tf.get_variable("b2", [1]) 
# 需要定义模型: y=xw+b 
h = tf.matmul(x, w1) + b1 
h = tf.nn.relu(h)
y = tf.matmul(h, w2) + b2
-----------------------
# 将全连接层封装成高层次API
# 由此可以看出，全连接层只是计算出了目标函数
"""
h = tf.layers.dense(
    x, 
    1000, 
    activation=tf.nn.relu)
y = tf.layers.dense(
    h, 
    1, 
    activation=None)

"""
# 定义loss函数
loss = (y-d)**2
loss = tf_FC.reduce_mean(loss)
"""
loss = tf.reduce_mean(tf.square(y-d))

"""
# 定义优化器及梯度，等同于optimizer.minimize(loss)
grads = optimizer.compute_gradients(
    loss, 
    [w, b]) 
# 执行：w=w-eta*dw 
train_step = optimizer.apply_gradients(grads)
---------------------------------------------
# 一个寻找全局最优点的优化算法，引入了二次方梯度校正。
# optimizer = tf_FC.train.AdamOptimizer(0.01)
"""
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(loss) 


# 输入样本进行训练
# 定义会话 
sess = tf.Session() 
sess.run(tf.global_variables_initializer())

"""
# 画出计算图，保存在graph_path目录中
# 运行完后，在Terminal输入tensorboard --logdir = graph_path
# 在弹出的url中查看 
"""
tf.summary.FileWriter("graph", graph=sess.graph)

# 读取训练数据
import numpy as np
# file = np.load("homework.npz")
# data_x = file['X']
# data_d = file['d']
data_x = np.random.random([3000, 1]) * 6 
data_d = np.sin(data_x)

for step in range(1000): 
    idx = np.random.randint(0, 1000, [32]) 
    inx = data_x[idx] 
    ind = data_d[idx] 
    st, ls = sess.run(
        [train_step, loss], 
        feed_dict={
            x: inx,
            d: ind
        }
    )
    print(ls)  

pred_y = sess.run(y, feed_dict={x: data_x})

import matplotlib.pyplot as plt
plt.scatter(data_x[:, 0], data_d[:, 0], c="y") 
plt.scatter(data_x[:, 0], pred_y[:, 0], c="r") 
plt.show()
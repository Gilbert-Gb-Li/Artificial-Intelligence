#!/usr/bin/env python  
import tensorflow as tf
import numpy as np  
"""
 tf 实现线性回归
 """

# 定义可接接收变量
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)  
# 定义变量
w = tf.Variable(0.0, name="weights")
b = tf.Variable(0.0, name="biases")

# 目标函数
def model(X, w, b):  
    return tf.multiply(X, w) + b
  
y_model = model(X, w, b)

# loss
cost = tf.square(Y - y_model)

# diff
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# run
with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)

    trX = np.linspace(-1, 1, 101)
    trY = 2 * trX + np.ones(*trX.shape) * 4 + np.random.randn(*trX.shape) * 0.03

    for i in range(100):
        for (x, y) in zip(trX, trY):  
            sess.run(train_op, feed_dict={X: x, Y: y})  
  
    # print weight  
    print(sess.run(w))
    # print bias  
    print(sess.run(b))



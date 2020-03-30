import tensorflow as tf
import numpy as np
"""
# 显式的定义计算图
# 以计算图为执行基本单元
"""
g = tf.graph()
with g.as_default():
    a1 = tf.constant(np.ones([4, 4]))
    a2 = tf.constant(np.ones([4, 4]))
    # 矩阵点积
    a1_dot_a2 = tf.matmul(a1, a2)
"""
# 画出计算图，保存在graph目录中
# 运行完后，在控制台输入tensorboard --logdir=graph
# 在弹出的url中查看 
"""
tf.summary.FileWriter("graph", graph=g)
''' 显示的指定发送给session的计算图'''
sess = tf.Session(graph=g)
print(sess.run(a1_dot_a2))
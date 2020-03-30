import tensorflow as tf
import numpy as np

a1 = tf.constant(np.ones([4, 4]) * 2)
a2 = tf.constant(np.ones([4, 4]))
b1 = tf.Variable(a1)
# 定义变量，另一种形式
# name 为计算图中节点的名字
# b2 = tf_FC.get_variable("name",[4,4])
b2 = tf.Variable(np.ones([4, 4]))
# 通过节点名称方式定义变量，当名不存在就创建
b2 = tf.get_variable(np.ones([4, 4], "name"))
# 定义placeholder，不断从外界接受变量
c2 = tf.placeholder(dtype=tf.float64, shape=[4, 4])

a1_elementwise_a2 = a1 * a2
a1_dot_a2 = tf.matmul(a1, a2)

b1_elementwise_b2 = b1 * b2
b1_dot_b2 = tf.matmul(b1, b2)

c2_dot_b2 = tf.matmul(c2, b2)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# feed [feed_dict] 来传入变量
print(sess.run(c2_dot_b2, feed_dict={c2: np.zeros([4, 4])}))

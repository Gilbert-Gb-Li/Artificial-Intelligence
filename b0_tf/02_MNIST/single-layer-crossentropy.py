#by cangye@hotmail.com
#引入库
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

"""
交叉熵损失函数
=================
"""

# 获取数据
mnist = input_data.read_data_sets("data/", one_hot=True)
# 构建网络模型
# x，label分别为图形数据和标签数据
x = tf.placeholder(tf.float32, [None, 784])
label = tf.placeholder(tf.float32, [None, 10])
# 构建单层网络中的权值和偏置
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
"""
=================================================
1.1 在使用交叉熵作为loss函数时，最后一层不加入激活函数
logits.shape: (100,10) 
logits[0]: [-∞, +∞]
  [-5.39518356e-01 -1.63998723e-01  1.44882798e-02  5.80523014e-02
  -2.73873508e-01  3.14608037e-01  2.54374415e-01 -2.49788761e-02
   3.76594245e-01 -1.34752005e-01]
==================================================
"""
logits = tf.matmul(x, W) + b

"""
1.2 输出需转换为概率分布
========================================================================
# 手工实现： 
# tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
----------------------- 
prod.shape: (100,10)    # 每次计算的样本为100个，每个样本输出为10个元素的向量
prod[0]: [0, 1]
  [2.24622205e-08 9.75589454e-01 7.35127978e-05 1.36152923e-03
  4.29530161e-07 1.79347146e-04 1.50707041e-04 5.79347261e-06
  2.25301385e-02 1.08987333e-04]
========================================================================
"""
prob = tf.nn.softmax(logits)

"""
## +===============================+ ##
# 1.3 使用交叉熵作为损失函数
# 手工实现
# - label * tf.log(prob) 交叉熵
--------------------------------
loss = tf.reduce_mean(tf.reduce_sum(- label * tf.log(prob), axis=1))
loss.shape: (100, 10)
loss[0]: [0, 0,...1,...0]
  [-0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
  -0.00000000e+00  1.20224737e-01 -0.00000000e+00 -0.00000000e+00
  -0.00000000e+00 -0.00000000e+00]
## +===============================+ ##
"""
# 可以使用tf中的实现
loss = tf.losses.softmax_cross_entropy(label, logits)


# 用梯度迭代算法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 用于验证
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 定义会话
sess = tf.Session()
# 初始化所有变量
sess.run(tf.global_variables_initializer())
# 迭代过程
train_writer = tf.summary.FileWriter("mnist-logdir", sess.graph)
for itr in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, label: batch_ys})
    if itr % 10 == 0:
        print("step:%6d  accuracy:"%itr, sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        label: mnist.test.labels}))

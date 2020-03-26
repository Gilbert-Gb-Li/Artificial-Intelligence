# by cangye@hotmail.com
"""
贷款欺诈案例最基础实现
"""

import pandas as pd
import tensorflow as tf
import numpy as np
'''1.读取数据'''
data = pd.read_csv("data/creditcard.csv")
class1 = data[data.Class == 0]
class2 = data[data.Class == 1]
print(len(class1))
print(len(class2))
print(np.shape(class1.values))

'''2.获取数据值'''
data1 = class1.values
data2 = class2.values

'''3.定义神经网络'''
'''3.1定义接收变量'''
x = tf.placeholder(tf.float32, [None, 28], name="input_x")
label = tf.placeholder(tf.float32, [None, 2], name="input_y")
'''3.2定义权值'''
W1 = tf.get_variable('W1',
                    [28, 28], 
                    dtype=tf.float32, 
                    initializer=tf.truncated_normal_initializer(stddev=0.1))
b1 = tf.get_variable('b1', 
                    [28], 
                    dtype=tf.float32, 
                    initializer=tf.constant_initializer(0))
h1 = tf.nn.sigmoid(tf.matmul(x, W1)+b1)
'''3.3定义第二层网络'''
W2 = tf.get_variable('W2',
                    [28, 2], 
                    dtype=tf.float32, 
                    initializer=tf.truncated_normal_initializer(stddev=0.1))
b2 = tf.get_variable('b2', 
                    [2], 
                    dtype=tf.float32, 
                    initializer=tf.constant_initializer(0))
h2 = tf.matmul(h1, W2)+b2
y = tf.nn.sigmoid(h2)

"""
4.定义loss函数
"""
loss = tf.reduce_mean(tf.square(y-label))

"""
5.定义准确率计算
tf.argmax()返回最大数值的下标
axis=0 按列找 
axis=1 按行找 
"""
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''6.定义导函数'''
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

'''开始迭代过程'''
for itr in range(300):
    idx1 = np.random.randint(284000)
    idx2 = np.random.randint(400)

    # 正负样本各采样25个
    # 使每次训练用的正负样本基本均衡
    feedx = np.concatenate([data1[idx1:idx1+25, 1:29],
                            data2[idx2:idx2+25, 1:29]])
    feedy = np.zeros([50, 2])
    feedy[:25, 0] = 1
    feedy[25:, 1] = 1
    
    sess.run(train_step, feed_dict={x: feedx, label: feedy})
    if itr % 1 == 0:
        feedx = np.concatenate([data1[3000:3000+400, 1:29],
                                data2[:400, 1:29]])
        feedy = np.zeros([800, 2])
        feedy[:400, 0] = 1
        feedy[400:, 1] = 1
        print("step:%6d  accuracy:"%itr, 100*sess.run(accuracy, feed_dict={x: feedx,
                                        label: feedy}))
import matplotlib.pyplot as plt
plt.plot(sess.run(W1.value()))
plt.show()
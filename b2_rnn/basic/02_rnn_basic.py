# !/usr/bin/python
# -*- coding: utf-8 -*-
# cangye@hotmail.com
"""
=====================
rnn 【循环神经网络】
=====================
rnn 对全链接网络的扩展
1. 单层循环网络
2. 多层循环网络
"""

import numpy as np
import tensorflow as tf

'''
1.
定义参数
'''
np.random.seed(0)
'''样本数量'''
batch_size = 12
'''序列长度，类似于表格数据的特征'''
len_sequence = 100
'''每个元素降维后的向量大小'''
embed_size = 6
'''【隐藏层】输出y_t的维度 '''
num_units = 6


""" 
2.
输入数据
为Embedding后的数据
"""
inputx = np.random.random([batch_size, len_sequence, embed_size])
""" np 转换为tf格式 """
indata = tf.constant(inputx)


"""
3.
RNN模型调用
"""

"""单层循环网络"""
# cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden)
# cell = tf.nn.rnn_cell.BasicLSTMCell(6, state_is_tuple=True)
"""多层循环网络"""
cell = tf.nn.rnn_cell.MultiRNNCell([
    tf.nn.rnn_cell.BasicRNNCell(num_units),
    tf.nn.rnn_cell.BasicRNNCell(num_units)
])
"""初始状态向量， yt"""
state = cell.zero_state(batch_size, tf.float64)


"""
4.
训练, 循环迭代时间步, 结果输出
outputs.shape: (len_sequence, batch_size, embedding_size)
    shape通过参数指定也可以是: (batch_size, len_sequence, embedding_size)
outputs每个元素为: [batch_size, embedding_size]
state[0].shape: [batch_size, embedding_size]
state[1].shape: [batch_size, embedding_size]
"""
outputs = []
for time_step in range(len_sequence):
    # if time_step > 0: tf.get_variable_scope().reuse_variables()
    """
    cell_output, 最终输出的状态
    state, 包含所有层输出的状态
    """
    (cell_output, state) = cell(indata[:, time_step, :], state)
    outputs.append(cell_output)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for idx, itr in enumerate(sess.run(outputs)):
    # print("step%d:" % idx, itr)
    pass
print(np.array(outputs).shape)
sess.close()

# """
# 使用numpy来模拟tf的计算过程
# ========================
# 1. 获取所有可训练变量
# """
# varlist = tf.trainable_variables()
# w, b = sess.run(varlist)
# # print(w, b)
# """
# 2. 激活函数：
#     tanh，双曲正切
#     Relu 函数值是没有上限，循环过程中会出现指数级增长
# """
# s = np.zeros([1, 6])
# for time_step in range(10):
#     x = inputx[:, time_step, :]
#     x = np.concatenate([x, s], axis=1)
#     s = np.tanh(np.dot(x, w) + b)
#     print("step%d" % time_step, s)

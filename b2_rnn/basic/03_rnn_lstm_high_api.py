#!/usr/bin/python
# -*- coding: utf-8 -*-
# cangye@hotmail.com
"""
=====================
rnn 单元的使用
建立序列间的关联关系
-----------------
代码内容：
1. LSTM 循环网络
    - 含有记忆单元的循环网络，记录某些元素的重要程度，相当于加权
    - BasicLSTMCell -> call()
2. 多层循环网络
3. 循环网络tf高级API
====================="""

import numpy as np
import tensorflow as tf

np.random.seed(0)
indata = tf.constant(np.random.random([1, 10, 6]))


"""lstm 循环网络"""
cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=6, state_is_tuple=True)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2, state_is_tuple=True)

"""
训练
参数：
    cell, 神经网络RNN
    indata, 输入
    sequence_length, 指定计算的序列长度
    time_major, 指定序列输入的格式， true => [max_time, batch_size, depth]； false => [batch_size, max_time, depth]
    state, 可以不指定 [state = cell.zero_state(1, tf.float64)]
    dtype, 类型
输出：
outputs: 最后一层的输出, 默认shape: (Batch, Term, Channel)
last_state: 每一层的最后一个step的输出最终的状态, states中的最后一个array正好是outputs的最后那个step的输出
    一般情况下state的形状为 [layers, state]
    但当输入的cell为BasicLSTMCell时，state的形状为[layers, c, h]，
"""
outputs, last_state = tf.nn.dynamic_rnn(cell, indata, dtype=tf.float64, sequence_length=[100])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
res = sess.run({'out': outputs, 'state': last_state})
print('out: \n', res['out'])
print('state: \n', res['state'])
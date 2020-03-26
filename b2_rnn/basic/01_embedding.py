"""
文本向量化及降维
================
Embedding
将字符ID转换为向量
"""

import numpy as np
import tensorflow as tf


"""
1. 字符串转ID
-------------------
参数解释：
    str = "向 量 化"
    input_data: [[31, 92, 103]]
        - 事先定义了字典， 假设“向 量 化” 三个字在字典中对应编号是 31, 92, 10；
        - get_variable 与 constant只能输入一维向量
           Variable可以输入多维等长向量
        - 输入向量ID必须在降维矩阵范围之内
            如，降维矩阵为[100, 8], 则 ID为103会报错 indices[0,2] = 103 is not in [0, 100)
"""
input_data = tf.Variable([[31, 92, 103], [27, 43, 9]], tf.int32)


"""
2. 定义降维矩阵： [2000, 128] => [in, out]
    2000，为输入向量长度，即文本长度
    128，输出维度的大小
"""
w = tf.constant(np.random.normal(0, 1, [2000, 128]))


"""
3. embedding_lookup
参数:
    W, 降维矩阵
        shape: (字典长度， 降维后长度)
    X, 待编码数据
        shape: (样本个数，样本序列长度)
输出: 降维后的矩阵
    shape: (BatchSize, sequenceLen, embedding_size) => [ [ [...] ] ]
    BatchSize: 样本，语句的个数
    SequenceLen: 序列长度，类似于样本特征数量， 即每个语句字数
    embedding_size： 每个字的向量长度， 一遍取值为64, 128, 256, 512
------------------------
认为每个词按时间顺序输入 x1, x2, x3 ...
    x1 = inputs[:, 0, :] -> 所有样本的第0个输入
"""
inputs = tf.nn.embedding_lookup(w, input_data)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run(inputs))

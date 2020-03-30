import tensorflow as tf  
"""
==============
文本分词网络模型
==============
基于CNN的文本分词 
"""

# files = open("data/poetry.txt", encoding="utf-8")
# data = files.read()
# char = set(data)

'''-----------
定义参数
--------------'''
# 字符个数
n_words = 5387 
# 类别个数
n_class = 4 
# 批尺寸
batch_size = 32  
# 输入序列长度，时间步
seq_len = 100
# 字符向量长度
embedding_size = 64 
# 循环神经网络层
n_layers = 2
# 隐藏层向量长度
n_hidden = 128

'''-------------
确定神经网络的输入
----------------'''
inputs_id = tf.placeholder(tf.int32, [batch_size, seq_len]) 
target_id = tf.placeholder(tf.int32, [batch_size, seq_len])  
weights = tf.placeholder(tf.float32, [batch_size, seq_len])
emb_w = tf.get_variable("emb_w", [n_words, embedding_size])

'''-----------
CNN 网络模型， tf API
重要参数：
    padding, 必须为"same"，保证序列长度
    stride, 必须为1，保证序列间关系
--------------'''
net = tf.nn.embedding_lookup(emb_w, inputs_id)
net = tf.layers.conv1d(net, n_hidden, 3, activation=tf.nn.relu, padding="same")
net = tf.layers.max_pooling1d(net, 3, 1)
net = tf.layers.conv1d(net, n_hidden, 3, activation=tf.nn.relu, padding="same")
net = tf.layers.conv1d(net, n_hidden, 3, activation=tf.nn.relu, padding="same")
net = tf.layers.conv1d(net, n_hidden, 3, activation=tf.nn.relu, padding="same")

'''--------
以下参考 RNN
-----------'''
logit = tf.layers.dense(net, n_class) 
loss = tf.contrib.seq2seq.sequence_loss(logit, target_id, weights)
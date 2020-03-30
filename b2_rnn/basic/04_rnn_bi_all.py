"""==================
文本分词网络模型
--------------
- sequence loss
- 双向LSTMCell RNN
====================="""
import tensorflow as tf
# files = open("data/poetry.txt", encoding="utf-8")
# data = files.read()
# char = set(data)

"""------------
# 定义参数
------------"""
# 字符个数
n_words = 5387 
# 类别 B M E S， 个数为4
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

'''---------------------
1. 确定神经网络的输入
---------------------'''
inputs_id = tf.placeholder(tf.int32, [batch_size, seq_len])

''' ##
target.shape:[batch_size, seq_len]
每个时间步都要输出
## '''
target_id = tf.placeholder(tf.int32, [batch_size, seq_len])
weights = tf.placeholder(tf.float32, [batch_size, seq_len])

'''---------------------
2. Embedding 输入数据
------------------------'''

''' 2.1 前向输入数据 '''
emb_w = tf.get_variable("emb_w", [n_words, embedding_size]) 
fw_input = tf.nn.embedding_lookup(emb_w, inputs_id)
''' 2.2 反向输入数据 '''
bw_input = tf.reverse(fw_input, axis=1)

'''----------------
3. RNN 循环网络
-------------------'''

'''# 定义两层循环网络单元：前后向定义方法相同 #'''
cell_fw = tf.nn.rnn_cell.MultiRNNCell(
    [tf.nn.rnn_cell.LSTMCell(n_hidden) for layer in range(n_layers)]
) 
cell_bw = tf.nn.rnn_cell.MultiRNNCell(
    [tf.nn.rnn_cell.LSTMCell(n_hidden) for layer in range(n_layers)]
)

''' ##
3.1 前向embed数据输入前向网络
3.2 后向embed数据输入后向网络
3.3 将后向RNN网络输出翻转
3.4 两种输出简单粗暴连接
## '''
# outputs:[batch_size, seq_len, n_hidden]
fw_output, last_state = tf.nn.dynamic_rnn(
    cell_fw, fw_input, dtype=tf.float32
)
bw_output, last_state = tf.nn.dynamic_rnn(
    cell_bw, bw_input, dtype=tf.float32
)
bw_output = tf.reverse(bw_output, axis=1)

''' # tf高阶API # '''
# (fw_output, bw_output), (s1, s2) = tf.nn.bidirectional_dynamic_rnn(
#     cell_fw,
#     cell_bw,
#     fw_input
# )
net = tf.concat([fw_output, bw_output], axis=2)

'''----------------------
4. 全连接层进行分类训练
-----------------------'''

'''# 每个时间步均是分类问题 #'''
logits = tf.layers.dense(net, n_class)

''' ##
定义连续交叉熵作为loss.
乘以sequence mask矩阵，计算每步的loss加权求平均。
重要参数：
    logits, 预测输出, 无需转softmax [batch_size, seq_len, n_class]
    targets, int值标签, 无需转one-hot
    weight, 每步预测的权重
## '''

loss = tf.contrib.seq2seq.sequence_loss(logits, target_id, weights)
'''-----------------------
5. 建立session，优化执行(略)
--------------------------'''
sess = tf.Session()
sess.close()
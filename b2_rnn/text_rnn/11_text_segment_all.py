"""
==============
文本分词网络模型
==============
基于CNN的文本分词
"""

import os
import numpy as np

class Data:

    def __init__(self, file_name="../data/pku_training.utf8"):
        base_dir = "model"
        files = open(file_name, "r", encoding="utf-8")
        datas = files.read().replace("\n", " ")
        data_list = datas.split(' ')
        def label(txt):
            len_txt = len(txt)
            if len_txt == 1:
                ret = 's'
            elif len_txt == 2:
                ret = 'be'
            elif len_txt > 2:
                mid = 'm' * (len_txt-2)
                ret = 'b' + mid + 'e'
            else:
                ret = ''
            return ret 
        data_labl = [label(itr) for itr in data_list]
        datas = ''.join(data_list)
        label = ''.join(data_labl)
        words_set = set(datas)
        label2id = {'s': 0, 'b': 1, 'm': 2, 'e': 3}
        self.id2label = {0: 's', 1: 'b', 2: 'm', 3: 'e'}
        if not os.path.exists(os.path.join(base_dir, "words2id")):
            self.word2id = dict(zip(words_set, range(len(words_set))))
            with open(os.path.join(base_dir, "words2id"), "w", encoding="utf-8") as f:
                f.write(str(self.word2id))
        else:
            with open(os.path.join(base_dir, "words2id"), "r", encoding="utf-8") as f:
                self.word2id = eval(f.read())
        if not os.path.exists(os.path.join(base_dir, "id2word")):
            self.id2word = dict()
            for itr in self.word2id:
                self.id2word[self.word2id[itr]] = itr
            with open(os.path.join(base_dir, "id2word"), "w", encoding="utf-8") as f:
                f.write(str(self.id2word))
        else:
            with open(os.path.join(base_dir, "id2word"), "r", encoding="utf-8") as f:
                self.id2word = eval(f.read())
        self.words_len = len(self.word2id)
        self.data_ids = np.array([self.word2id.get(itr, 0) for itr in datas])
        self.label_ids = np.array([label2id.get(itr, 0) for itr in label])
        self.seqlen = len(self.data_ids)
        np.savez("words_seg.npz", data=self.data_ids, label=self.label_ids)

    def next_batch(self, batch_size=32):
        length = 50
        x = np.zeros([batch_size, length])
        d = np.zeros([batch_size, length])
        for itr in range(batch_size):
            idx = np.random.randint(0, self.seqlen-length)
            x[itr, :] = self.data_ids[idx:idx+length]
            d[itr, :] = self.label_ids[idx:idx+length] 
        return x, d 
    def w2i(self, txt):
        data = [self.word2id.get(itr, 0) for itr in txt] 
        data = np.array(data) 
        data = np.expand_dims(data, 0)
        return data
    def i2l(self, data):
        return ''.join([self.id2label.get(itr, 's') for itr in data])

# 输入：今天的天气不错
# 标签：B E S B M M E 
data_tools = Data()
import tensorflow as tf 
# 定义输入：任意长度
Times = 50
BatchSize = 32
n_words = data_tools.words_len 
embedding_size = 128 
n_hidden = 128  
n_layers = 2
n_class = 4

inputs_ID = tf.placeholder(
    tf.int32,   # 词ID为整形数字
    [None, None])
labels_ID = tf.placeholder(
    tf.int32,   # 类ID
    [BatchSize, Times]
)
# 消除补0部分的影响
mask = tf.placeholder(
    tf.float32, 
    [BatchSize, Times]
)
# Embeddings
# 定义降维矩阵
embedding_w = tf.get_variable(
    "embw",     # 变量名
    [n_words, embedding_size]
)
# emb_inputs:Embedding后的向量
emb_inputs = tf.nn.embedding_lookup(
    embedding_w,
    inputs_ID
)

# 定义2层RNN网络单元
rnn_fn = tf.nn.rnn_cell.BasicLSTMCell
cell = tf.nn.rnn_cell.MultiRNNCell(
    [rnn_fn(n_hidden) for _ in range(n_layers)])
# outputs循环神经网络的输出:[BatchSize, Times, n_hidden]
outputs, last_state = tf.nn.dynamic_rnn(
    cell,   # RNN单元
    emb_inputs,     # embedding_inputs
    dtype=tf.float32
    )

cell = tf.nn.rnn_cell.MultiRNNCell(
    [rnn_fn(n_hidden) for _ in range(n_layers)])

logit = tf.layers.dense(outputs, n_class, activation=None)

loss = tf.contrib.seq2seq.sequence_loss(
    logit, 
    labels_ID,
    mask
)

step = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess = tf.Session() 
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()


for itr in range(2000):
    inx, iny = data_tools.next_batch(BatchSize)
    ls, _ = sess.run([loss, step], 
    feed_dict={
        inputs_ID: inx,
        labels_ID: iny,
        mask: np.ones([BatchSize, Times])
    })
    if itr % 10 == 0:
        saver.save(sess, "modelx/seg", itr)
        dt = data_tools.w2i("实现祖国的完全统一")
        ids = sess.run(tf.argmax(logit, 2), feed_dict={inputs_ID:dt})
        print(data_tools.i2l(ids[0]))
        print(ls)
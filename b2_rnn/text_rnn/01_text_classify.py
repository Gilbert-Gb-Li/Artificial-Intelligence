import tensorflow as tf  
import numpy as np
from configs import d_path

def getWordsFromFile(path):
    """
    读取文件返回数据和标签
    :param path:
    :return:
    """
    target = []
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            target.append(line[0: 2])
            data.append(line[2: -1])
    return data, target

def getWordsMap(data):

    """
    将数据驱虫转换为字典
    data is a list
    """
    words = set(''.join(data))
    n_words = len(words)
    d_words = dict(zip(words, range(len(words))))
    return d_words, n_words

def makeWordId(dic, data):

    """
    将数据根据字典转换为ID
    预测数据集有可能出现字典中没有的字，加入到其他，即4789
    data is a list
    """
    ids = []
    for itr in data:
        id = []
        for s in itr:
            try:
                id.append(dic[s])
            except:
                id.append(4789)
        ids.append(id)
    return ids


def targetProcess(target):
    """
    标签处理，转ID
    :param target:
    :return:
    """
    s_target = set(target)
    n_target = len(s_target)
    d_target = {item: idx for idx, item in enumerate(s_target)}
    id_target = [d_target[itr] for itr in target]
    return id_target, n_target


class RNNModel:

    """
    ===============
    1. 定义参数
    ===============
    """
    def __init__(self, n_words, n_class, n_sample):
        self.n_words = n_words  # n_words: 字典中字个数， len(words)
        self.n_class = n_class
        self.batch_size = 32
        self.seq_len = 100  # 输入序列长度，时间步；此处每个样本取前100个字做输入
        self.embedding_size = 64    # embedding后每个字的向量长度
        self.n_layers = 2
        self.n_hidden = 128
        self.sess = tf.Session()
        self.saver = tf.train.Saver()

    """
    =================
    2. 定义双层RNN循环网络
    =================
    """
    def __rnn(self, id_inputs):

        """
        2.1 定义神经网络的输入
            tf.placeholder 也可以直接使用变量代替
        """
        with tf.variable_scope('V1'):
            # inputs_id = tf.placeholder(tf.int32, [self.batch_size, self.seq_len])
            # target_id = tf.placeholder(tf.int32, [self.batch_size])
            self.emb_w = tf.get_variable("emb_w", [n_words, self.embedding_size])
            emb_data = tf.nn.embedding_lookup(self.emb_w, id_inputs)

            """2.2. 定义多层RNN：两层循环网络单元 """
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.LSTMCell(self.n_hidden) for layer in range(self.n_layers)]
            )
            """2.3 将Embedding后的向量输入多层RNN中"""
            # outputs: [batch_size, seq_len, n_hidden]
            outputs, last_state = tf.nn.dynamic_rnn(
                cell, emb_data, dtype=tf.float32
            )
            """2.4 获取最后时间步输出"""
            net = outputs[:, -1, :]
            return net

    """ 
    ==================
    3. 定义全连接层，转换为分类问题
    ==================
        输入：
            rnn后最后一个时间步 [batch_size, embedding_size]
            标签ID
        输出：
            logit: softmax概率 [batch_size, n_class]
            loss: 交叉熵损失函数
    """
    def __fc(self, net, target_id):
        net = tf.layers.dense(net, self.n_class)
        '''# tensor flow one hot '''
        target_one_hot = tf.one_hot(target_id, self.n_class)

        '''# 转概率并计算交叉熵 '''
        logit = tf.nn.softmax(net)
        loss = tf.reduce_sum(-target_one_hot * tf.log(logit), axis=1)
        loss = tf.reduce_mean(loss)
        # loss = tf.losses.softmax_cross_entropy(label, logits)
        '''# 优化函数 '''
        step = tf.train.AdamOptimizer(0.01).minimize(loss)
        return step, loss, logit

    def train(self, ids, target):
        net = self.__rnn(ids)
        step, loss, logit = self.__fc(net, target)
        """执行计算"""

        self.sess.run(tf.global_variables_initializer())
        for itr in range(1):
            result = self.sess.run({'step': step, 'loss': loss, 'logit': logit})
            print('step: {}, logit: {}'.format(itr, result['logit'].shape))
        self.saver.save(self.sess, 'my-poetry', global_step=0)

    def predict(self, ids_test, y_test):
        """
        预测
        :param ids_test: 数据的ID
        :param y_test: 标签的ID
        :return: softmax 概率
        """
        self.sess.run()
        with tf.variable_scope('V2'):
            emb_data = tf.nn.embedding_lookup(self.emb_w, ids_test)
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.LSTMCell(self.n_hidden) for layer in range(self.n_layers)]
            )
            outputs, last_state = tf.nn.dynamic_rnn(
                cell, emb_data, dtype=tf.float32
            )
            net = tf.layers.dense(outputs[:, -1, :], self.n_class)
            logit = tf.nn.softmax(net)
            # target_one_hot = tf.one_hot(y_test, self.n_class)
            return logit

    def __del__(self):
        self.sess.close()

if __name__ == '__main__':

    """训练数据预处理"""
    p_train = f"{d_path}texts/cnews/cnews.val.txt"
    data, target = getWordsFromFile(p_train)
    d_words, n_words = getWordsMap(data)
    ids = makeWordId(d_words, data)
    '''读取每段文本的前10个字作为训练预测数据，固定长度'''
    ids = [id[0: 10] for id in ids]
    id_target, n_target = targetProcess(target)

    """测试数据集处理"""
    p_test = f"{d_path}texts/cnews/cnews.test.txt"
    X_test, y_test = getWordsFromFile(p_test)
    ids_test = makeWordId(d_words, X_test)
    ids_test = [id[0: 10] for id in ids_test]
    id_y_test, n_test = targetProcess(y_test)

    """RNN"""
    rnn = RNNModel(n_words, n_target, len(data))
    rnn.train(ids, id_target)




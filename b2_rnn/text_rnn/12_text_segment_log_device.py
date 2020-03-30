import tensorflow as tf
import os
import numpy as np

"""
分词参考模型
"""


class Data:

    def __init__(self):
        """
        初始化数据读取
        """
        base_dir = "model"
        # 读取文件
        files = open("../data/pku_training.utf8", "r", encoding="utf-8")
        datas = files.read().replace("\n", " ")
        data_list = datas.split(' ')
        files.close()

        def label(txt):
            """
            将词数据转换为标签
            txt:文本词
            """
            len_txt = len(txt)
            if len_txt == 1:
                ret = 's'
            elif len_txt == 2:
                ret = 'be'
            elif len_txt > 2:
                mid = 'm' * (len_txt - 2)
                ret = 'b' + mid + 'e'
            else:
                ret = ''
            return ret
            # 设置文本标签

        data_label = [label(itr) for itr in data_list]
        # 将文本列表中的字连成字符串
        datas = ''.join(data_list)
        label = ''.join(data_label)
        words_set = set(datas)
        label2id = {'s': 0, 'b': 1, 'm': 2, 'e': 3}
        self.id2label = {0: 's', 1: 'b', 2: 'm', 3: 'e'}
        # 保存字典词转换为id  {'迈': 932, ...}
        if not os.path.exists(os.path.join(base_dir, "words2id")):
            self.word2id = dict(zip(words_set, range(len(words_set))))
            with open(os.path.join(base_dir, "words2id"), "w", encoding="utf-8") as f:
                f.write(str(self.word2id))
        else:
            with open(os.path.join(base_dir, "words2id"), "r", encoding="utf-8") as f:
                self.word2id = eval(f.read())
        # 保存字典id转换为词 {932: '迈', ...}
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
        # 将输入数据字符串、标签转成ID
        self.data_ids = np.array([self.word2id.get(itr, 0) for itr in datas])
        self.label_ids = np.array([label2id.get(itr, 0) for itr in label])
        self.seqlen = len(self.data_ids)
        np.savez("words_seg.npz", data=self.data_ids, label=self.label_ids)

    def next_batch(self, batch_size=32):
        """
        随机获取训练数据
        batch, 32; 固定长度为50
        """
        length = 50
        x = np.zeros([batch_size, length])
        d = np.zeros([batch_size, length])
        for itr in range(batch_size):
            idx = np.random.randint(0, self.seqlen - length)
            x[itr, :] = self.data_ids[idx:idx + length]
            d[itr, :] = self.label_ids[idx:idx + length]
        return x, d

    def w2i(self, txt):
        """
        txt:一段文本
        词转换为ID
        """
        data = [self.word2id.get(itr, 0) for itr in txt]
        data = np.array(data)
        data = np.expand_dims(data, 0)
        return data

    def i2l(self, data):
        """
        data:标签id序列
        输出标签转换
        """
        return ''.join([self.id2label.get(itr, 's') for itr in data])


class Model:
    """
    分词模型中使用双向RNN模型进行处理
    """

    def __init__(self, is_training=True):
        """
        初始化类
        """
        self.is_training = is_training
        self.data_tools = Data()
        self.words_len = self.data_tools.words_len
        self.build_model()
        self.init_sess(tf.train.latest_checkpoint("model/tfmodel"))

    def build_model(self):
        """
        构建计算图 
        """
        self.graph = tf.Graph()
        cell_fn = tf.nn.rnn_cell.BasicRNNCell()
        n_layer = 2
        hidden_size = 128
        with self.graph.as_default():
            # 输入文本序列
            self.inputs = tf.placeholder(tf.int32,
                                         [None, None],
                                         name="inputs")
            # 文本标注序列，用0-3数字表示
            self.target = tf.placeholder(tf.int32,
                                         [None, None],
                                         name="target")
            # 序列不等长在计算loss函数部分需要置0，也就是乘以mask
            self.mask = tf.placeholder(tf.float32,
                                       [None, None],
                                       name="mask")
            # 序列长度，在解码部分可能用到
            self.seqlen = tf.placeholder(tf.int32,
                                         [None],
                                         name="seqlen")
            if not self.is_training:
                self.seqlen = None

            # 定义双向RNN单元
            fw_cell_list = [cell_fn(hidden_size) for itr in range(n_layer)]
            bw_cell_list = [cell_fn(hidden_size) for itr in range(n_layer)]
            fw_cell = tf.nn.rnn_cell.MultiRNNCell(fw_cell_list)
            bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_cell_list)
            # Embedding
            emb_w = tf.get_variable("emb_w", [self.words_len, hidden_size])
            emb_input = tf.nn.embedding_lookup(emb_w, self.inputs)
            # 双向RNN， self.seqlen用于不同长度序列解码
            (fw_output, bw_output), state = tf.nn.bidirectional_dynamic_rnn(
                fw_cell,
                bw_cell,
                emb_input,
                sequence_length=None,
                dtype=tf.float32
            )
            # 将两个网络输出进行连接
            outputs = tf.concat([fw_output, bw_output], 2)
            self.logit = tf.layers.dense(outputs, 4)
            # 计算loss函数
            self.loss = tf.contrib.seq2seq.sequence_loss(
                self.logit,
                self.target,
                self.mask
            )
            self.loss = tf.reduce_mean(self.loss)
            # 优化
            self.step = tf.train.AdamOptimizer(0.01).minimize(self.loss)
            self.all_var = tf.global_variables()
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

    def init_sess(self, restore=None):
        """
        初始化会话
        """
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        self.sess = tf.Session(
            graph=self.graph,
            config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        # self.sess = tf.Session()
        self.sess.run(self.init)
        if restore is not None:
            self.saver.restore(self.sess, restore)

    def train(self):
        """
        训练函数
        """
        for itr in range(200000):
            inx, iny = self.data_tools.next_batch(32)
            loss, _ = self.sess.run([self.loss, self.step],
                                    feed_dict={
                                        self.inputs: inx,
                                        self.target: iny,
                                        self.seqlen: np.ones(32) * 50,
                                        self.mask: np.ones([32, 50])
                                    })
            if itr % 100 == 90:
                print(itr, loss)
                self.saver.save(self.sess, "model/tfmodel-basic/segnet")
                self.predict(
                    "实现祖国完全统一，是海内外一切爱国的中华儿女的共同心愿。")
                self.predict(
                    "瓜子二手车直卖网，没有中间商赚差价。车主多卖钱，买家少花钱")

    def predict(self, txt):
        """
        预测函数
        sess, 通过将sess设置为实例变量来在train与predict之间共享
        :param txt: 文本序列
        :return:
        """
        # [1, Length]
        inx = self.data_tools.w2i(txt)
        out = self.sess.run(tf.argmax(self.logit, 2),
                            feed_dict={self.inputs: inx})
        seq = []
        for a, b in zip(txt, self.data_tools.i2l(out[0])):
            seq.append(a + b)
        print('|'.join(seq))


model = Model()
model.train()

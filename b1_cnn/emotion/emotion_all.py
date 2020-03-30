import tensorflow as tf
import numpy as np
import os
import configs


class Data:
    def __init__(self):
        p = configs.d_path
        if not os.path.exists("data.npz"):
            files = open(f"{p}emotion/fer2013.csv", "r", encoding="utf-8")
            datas = [itr.strip().split(',')
                     for itr in files.readlines()[1:]]
            labels = np.array([int(itr[0]) for itr in datas])
            images = np.array([[float(ii) for ii in itr[1].split(' ')]
                               for itr in datas
                               ])
            images = images - np.mean(images, 1, keepdims=True)
            images = images / (np.max(np.abs(images), 1, keepdims=True) + 1e-6)
            images = np.reshape(images, [-1, 48, 48, 1])
            np.savez("data.npz", images=images, labels=labels)
            self.images = images
            self.labels = labels
        else:
            files = np.load("data.npz")
            self.images = files['images']
            self.labels = files['labels']
        print("DataShape:", np.shape(self.images), np.shape(self.labels))
        print("max", np.max(self.images))
        self.seqlen = len(self.labels)

    def next_batch(self, batch_size=32):
        # numpy.random.randint(low, high=None, size=None, dtype='l')
        idx = np.random.randint(2000, self.seqlen, [batch_size])
        x = self.images[idx]
        d = self.labels[idx]
        return x, d

    def test_data(self):
        return self.images[:2000], self.labels[:2000]


class Model:
    """
    表情识别中使用VGGNet-16作为基本模型
    """

    def __init__(self, batch_size=None, is_training=True):
        """
        初始化类
        """
        self.batch_size = batch_size
        self.is_training = is_training
        self.build_model()
        self.init_sess()

    def build_model(self):
        """
        构建计算图
        """
        self.graph = tf.Graph()

        def block(net, n_conv, n_chl, blockID):
            """
            定义多个CNN组合单元
            """
            with tf.variable_scope("block%d" % blockID):
                for itr in range(n_conv):
                    net = tf.layers.conv2d(net,
                                           n_chl, 3,
                                           activation=tf.nn.relu,
                                           padding="same")
                    net = tf.layers.batch_normalization(net)
                net = tf.layers.max_pooling2d(net, 2, 2)
            return net

        with self.graph.as_default():
            # 人脸数据
            self.inputs = tf.placeholder(tf.float32,
                                         [self.batch_size, 48, 48, 1],
                                         name="inputs")
            inputs_img = tf.image.resize_images(
                self.inputs,
                [24, 24]
            )
            # 表情序列，用0-6数字表示
            self.target = tf.placeholder(tf.int32,
                                         [self.batch_size],
                                         name="target")
            target_onehot = tf.one_hot(self.target, 7)

            '''
            # VGG-16
            net = block(self.inputs, 2, 64, 1)
            net = block(net, 2, 128, 2)
            net = block(net, 2, 256, 3)
            net = block(net, 2, 512, 4)
            net = block(net, 2, 512, 5)
            net = tf.layers.flatten(net)
            net = tf.layers.dense(net, 4096, activation=tf.nn.relu)
            net = tf.layers.dense(net, 4096, activation=tf.nn.relu)
            '''

            net = tf.layers.conv2d(inputs_img, 32, 3)
            net = tf.layers.average_pooling2d(net, 2, 2)
            net = tf.layers.conv2d(net, 64, 3)
            net = tf.layers.conv2d(net, 64, 3)
            net = tf.layers.flatten(net)
            net = tf.layers.dense(net, 128, activation=tf.nn.relu)
            self.logits = tf.layers.dense(net, 7, activation=None)
            # 计算loss函数
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=target_onehot,
                logits=self.logits
            )
            accuracy = tf.equal(
                tf.argmax(self.logits, 1),
                tf.argmax(target_onehot, 1)
            )

            self.acc = tf.reduce_mean(tf.cast(accuracy, tf.float32))
            self.loss = tf.reduce_mean(self.loss)
            # 优化
            self.step = tf.train.AdamOptimizer(1e-5).minimize(self.loss)
            self.all_var = tf.global_variables()
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

    def init_sess(self, dirs="model/emotion"):
        """
        初始化会话
        """
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)
        files = tf.train.latest_checkpoint(dirs)
        if files != None:
            self.saver.restore(self.sess, files)

    def train(self):
        data = Data()
        for itr in range(10000):
            x, d = data.next_batch()
            ls, _ = self.sess.run(
                [self.loss, self.step],
                feed_dict={self.inputs: x,
                           self.target: d
                           }
            )
            if itr % 50 == 2:
                acc = self.sess.run(
                    self.acc,
                    feed_dict={self.inputs: data.images[:2000],
                               self.target: data.labels[:2000]
                               })
                print(f"Step{itr}, Loss:{ls}, Acc:{acc}")
                self.saver.save(self.sess, "model/emotion/emo")


if __name__ == '__main__':
    model = Model()
    model.train()

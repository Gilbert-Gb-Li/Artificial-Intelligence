# coding: utf-8
"""
逐行读取数据并处理
避免内存一次性加载
"""

import numpy as np
import os
import cv2


class Data:
    def __init__(self, data_dir):
        if not os.path.exists("data.npz"):
            # 数据读入
            files = open(data_dir, "r", encoding="utf-8")
            # 数据分隔符为逗号
            datas = [itr.strip().split(',')
                     for itr in files.readlines()[1:]]
            # 获取标签
            labels = np.array([int(itr[0]) for itr in datas])
            # 转换为图像
            images = np.array([[float(ii) for ii in itr[1].split(' ')]
                               for itr in datas
                               ])
            # 图像预处理
            images = images - np.mean(images, 1, keepdims=True)
            # 归一化处理：img/std(img)
            images = images / (np.std(images, 1, keepdims=True) + 1e-6)
            images = np.reshape(images, [-1, 48, 48, 1])
            # 保存文件备用
            np.savez("data.npz", images=images, labels=labels)
            self.images = images
            self.labels = labels
        else:
            # 如果文件存在则直接调用
            files = np.load("data.npz")
            self.images = files['images']
            self.labels = files['labels']
        print("DataShape:", np.shape(self.images), np.shape(self.labels))
        print("max", np.max(self.images))
        self.seqlen = len(self.labels)

    def next_batch(self, batch_size=32):
        """
        获取每次迭代的样本
        随机选择
        """
        idx = np.random.randint(0, self.seqlen, [batch_size])
        x = self.images[idx]
        d = self.labels[idx]
        return x, d

    def test_data(self):
        """
        选择前2000个样本作为测试集
        """
        return self.images[:2000], self.labels[:2000]


datatool = Data(data_dir="表情数据/fer2013.csv")
img = cv2.resize(datatool.images[0], (224, 224), interpolation=cv2.INTER_CUBIC)
cv2.imshow("a", img)
cv2.waitKey(0)

"""
inputs = tf.placeholder(tf.float32, 
                        [self.batch_size, 48, 48, 1],
                        name="inputs")
net = tf.image.resize_images(
                inputs, 
                [240, 240]
            )
"""

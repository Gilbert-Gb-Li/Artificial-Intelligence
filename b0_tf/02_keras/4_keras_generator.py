import keras
import math
import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


class DataGenerator(keras.utils.Sequence):

    def __init__(self, datas, batch_size=1, shuffle=True):
        self.batch_size = batch_size
        self.datas = datas
        '''样本的索引列表'''
        self.indexes = np.arange(len(self.datas))
        self.shuffle = shuffle

    def __len__(self):
        '''
        # 计算每一个epoch的迭代次数
        '''

        return math.ceil(len(self.datas) / float(self.batch_size))

    def __getitem__(self, index):
        '''
        # 生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        :param index: 当前训练步的初始索引
        :return:
        '''
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # 根据索引获取datas集合中的数据
        batch_datas = [self.datas[k] for k in batch_indexs]

        # 生成数据
        X, y = self.data_generation(batch_datas)

        return X, y

    def on_epoch_end(self):
        '''
        # 每个epoch结束时执行的操作
        # 此处在每一次epoch结束是否需要进行一次随机，重新随机一下index
        '''
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_datas):
        '''
        自定义函数
        '''
        images = []
        labels = []

        # 生成数据
        for i, data in enumerate(batch_datas):
            # x_train数据
            image = cv2.imread(data)
            image = list(image)
            images.append(image)
            # y_train数据
            right = data.rfind("\\", 0)
            left = data.rfind("\\", 0, right) + 1
            class_name = data[left:right]
            if class_name == "dog":
                labels.append([0, 1])
            else:
                labels.append([1, 0])
        # 如果为多输出模型，Y的格式要变一下，外层list格式包裹numpy格式是list[numpy_out1,numpy_out2,numpy_out3]
        return np.array(images), np.array(labels)


if __name__ == '__main__':
    # 读取样本名称，然后根据样本名称去读取数据
    class_num = 0
    train_datas = []
    for file in os.listdir("D:/xxx"):
        file_path = os.path.join("D:/xxx", file)
        if os.path.isdir(file_path):
            class_num = class_num + 1
            for sub_file in os.listdir(file_path):
                train_datas.append(os.path.join(file_path, sub_file))

    # 数据生成器
    training_generator = DataGenerator(train_datas)

    # 构建网络
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=784))
    model.add(Dense(units=2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    '''
    参数：
        validation_steps: 仅当 validation_data 是一个生成器时才可用。
            在停止前 generator 生成的总步数（样本批数）。 对于 Sequence，它是可选的：如果未指定，将使用 len(generator) 作为步数。
        class_weight: 可选的将类索引（整数）映射到权重（浮点）值的字典，用于加权损失函数（仅在训练期间）。 
            这可以用来告诉模型「更多地关注」来自代表性不足的类的样本。
        max_queue_size: 整数。生成器队列的最大尺寸。 如未指定，max_queue_size 将默认为 10。
        workers: 整数。使用的最大进程数量，如果使用基于进程的多线程，默认为 1。如果为0，将在主线程上执行生成器。
        use_multiprocessing: 布尔值。如果 True，则使用基于进程的多线程。
            如未指定， use_multiprocessing 将默认为 False。 
            请注意，由于此实现依赖于多进程，所以不应将不可传递的参数传递给生成器，因为它们不能被轻易地传递给子进程。
        shuffle: 是否在每轮迭代之前打乱 batch 的顺序。 只能与 Sequence (keras.utils.Sequence) 实例同用。
        initial_epoch: 开始训练的轮次（有助于恢复之前的训练）
    '''
    model.fit_generator(training_generator,
                        epochs=50,
                        max_queue_size=10,
                        workers=1)

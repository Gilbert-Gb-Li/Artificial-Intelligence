from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from configs import d_path
"""==============
keras 卷积神经网络
================="""

batch_size = 16
num_classes = 10
epochs = 2
img_rows, img_cols = 28, 28

'''-------------------
1. 导入数据
----------------------'''

# (x_train, y_train), (x_test, y_test) = mnist.load_data(path='G:/dl_data/mnist.npz')
# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     print("exec this method")
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)

x_train = np.load(f'{d_path}mnist/x_train.npy').reshape([-1, 28, 28, 1])
x_test = np.load(f'{d_path}mnist/x_test.npy').reshape([-1, 28, 28, 1])
y_train = np.load(f'{d_path}data/mnist/y_train.npy')
y_test = np.load(f'{d_path}data/mnist/y_test.npy')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# 归一化： Convert from [0, 255] -> [0.0, 1.0].
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)      # (60000, 28, 28, 1)
input_shape = (img_rows, img_cols, 1)

'''# 将整型标签转为one hot #'''
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# print(y_test[0])    # [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]

'''-------------
2. CNN神经网络
----------------'''

'''# 每个channel有自己的weight和bias。参数计算练习 #'''
model = Sequential()
''' #
卷积层
------------------------------------
add参数：layer，单层
Conv2D参数：
    filters：输出空间的维度（即卷积中滤波器的输出数量）
    input_shape：输入数据形状
        channels_first, (channels, rows, cols)
        channels_last, (rows, cols, channels)
    kernel_size：卷积核的宽度和长度
    strides：步长
    padding：补0策略， 'valid', 'same'
    activation：激活函数
    data_format：通道维的位置，'channels_last' 'channels_first'
输入：
    channels_first, (samples, channels, rows, cols)
    channels_last, (samples, rows, cols, channels)
输出shape
    channels_first：(samples, channels, rows, cols)
    channels_last： (samples, rows, cols, channels)
# '''
model.add(Conv2D(filters=32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,
                 kernel_initializer=keras.initializers.RandomNormal(stddev=0.01, seed=1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(2, (3, 3), strides=(2, 2), padding="valid", activation='relu'))
''' #
池化层
--------------------
MaxPooling2D参数：
    pool_size：在两个方向（竖直，水平）上的下采样因子
    strides：步长
    border_mode：‘valid’或者‘same’，卷积层padding
输出shape
    channels_first：(samples, channels, pooled_rows, pooled_cols)
    channels_last： (samples, pooled_rows, pooled_cols, channels)
# '''
model.add(MaxPooling2D(pool_size=(2, 2)))
'''# 可以加在各层中间 #'''
model.add(BatchNormalization())

'''----------------------
# 3. FC全连接层转化为分类问题
# -------------------------'''
# 输出使用全连接层
'''# 将多维数据展平为1维'''
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# 输出使用全连接层
model.add(Dense(num_classes, activation='softmax'))
# 打印统计信息
model.summary()
# 指定loss
#   sgd, 随机梯度下降算法，可以为SGD对象
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer="sgd",
              metrics=['accuracy'])
''' --
训练
    主要参数：
        epochs,
        batch_size
-- '''

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))     # 边训练边打印
score = model.evaluate(x_test, y_test, verbose=0)   # 准确度

'''# score: (loss, accuracy) #'''
print('Test loss:', score[0])
print('Test accuracy:', score[1])

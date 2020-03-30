
"""================================================
迁移学习
-------
# 基础：特征具有一定通用性
#      分阶段训练

1. 后期阶段训练时，有选择的屏蔽掉一部分， trainable = False
2. 使用model.layers
3. 使用tf Api opt.minimize(var_list=var) 的var_list参数
4. 使用成熟的网络： keras.io.applications
===================================================="""

from __future__ import print_function
import tensorflow as tf
import datetime
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

'''------------
参数定义
---------------'''
now = datetime.datetime.now

batch_size = 128
num_classes = 5
epochs = 1

img_rows, img_cols = 28, 28
filters = 32
pool_size = 2
kernel_size = 3

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)


def train_model(model, train, test, num_classes):
    '''# 训练方法 #'''
    x_train = train[0].reshape((train[0].shape[0],) + input_shape)
    x_test = test[0].reshape((test[0].shape[0],) + input_shape)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # one hot
    y_train = keras.utils.to_categorical(train[1], num_classes)
    y_test = keras.utils.to_categorical(test[1], num_classes)

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    t = now()
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    print('Training time: %s' % (now() - t))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


'''------------
1. 数据读取
---------------'''
(x_train, y_train), (x_test, y_test) = mnist.load_data(path='G:/dl_data/mnist.npz')

'''------------------
2. 切分数据集，模拟多次训练
---------------------'''
x_train_lt5 = x_train[y_train < 5]
y_train_lt5 = y_train[y_train < 5]
x_test_lt5 = x_test[y_test < 5]
y_test_lt5 = y_test[y_test < 5]

x_train_gte5 = x_train[y_train >= 5]
y_train_gte5 = y_train[y_train >= 5] - 5
x_test_gte5 = x_test[y_test >= 5]
y_test_gte5 = y_test[y_test >= 5] - 5

'''------------------
3. 创建模型
---------------------'''

'''# 定义CNN网络 #'''

feature_layers = [
    Conv2D(filters, kernel_size,
           padding='valid',
           input_shape=input_shape),
    Activation('relu'),
    Conv2D(filters, kernel_size),
    Activation('relu'),
    MaxPooling2D(pool_size=pool_size),
    Dropout(0.25),
    Flatten(),
]
'''# 全连接网络 #'''
classification_layers = [
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(num_classes),
    Activation('softmax')
]

'''-------------------
4.
迁移学习
----------------------'''

'''
# 4.1
# 使用keras trainable参数
# 创建model 
# 参数为1维list, 只能直接相加
'''
model = Sequential(feature_layers + classification_layers)


''' 
# *** 模拟不同阶段训练 ***
#     首先训练小于5的训练集
#     而后在训练大于5的训练集
'''
'''# 阶段1. 对小于5的数据集训练，标签为[0...4]'''
# train_model(model,
#             (x_train_lt5, y_train_lt5),
#             (x_test_lt5, y_test_lt5), num_classes)

''' ##
阶段2. 对大于5的数据集训练，屏蔽掉CNN层
    *** feature_layers中的权重不再训练 ***
## '''
for l in feature_layers:
    l.trainable = False     # 屏蔽掉

'''-----------------------
# 4.2
# model.layers, 获取其中的层
---------------------------'''
# for l in model.layers[:7]:
#     l.trainable = False

'''---------------
# 4.3
# 使用var_list参数
------------------'''
var = []
opt = tf.train.AdamOptimizer(1e-3)
step = opt.minimize(var_list=var)

model.summary()
train_model(model,
            (x_train_gte5, y_train_gte5),
            (x_test_gte5, y_test_gte5), num_classes)

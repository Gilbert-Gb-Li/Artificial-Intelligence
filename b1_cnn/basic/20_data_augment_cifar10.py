# coding: utf-8

"""=========
数据集扩充
============"""

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

batch_size = 32
num_classes = 10
epochs = 5
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# one hot
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer="sgd",
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

'''-------------
数据扩充
----------------'''
if not data_augmentation:
    print('Not using data augmentation.')
    '''------------------------
    fit，适用于小型，简单化的数据集
        1. 整个训练集会加载内存
        2. 无法数据增强
    ---------------------------'''
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    '''-----------------------------------------
    # 用以生成一个batch的图像数据，支持实时数据提升。
    # 训练时该函数会无限生成数据，直到达到规定的epoch次数为止。
    # https://keras-cn.readthedocs.io/en/latest/preprocessing/image/
    参数：
        rotation_range：整数，数据提升时图片随机转动的角度
        width_shift_range：浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
        height_shift_range：浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
        shear_range：浮点数，剪切强度（逆时针方向的剪切变换角度）
        zoom_range：浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
        horizontal_flip：布尔值，进行随机水平翻转
        vertical_flip：布尔值，进行随机竖直翻转
    --------------------------------------------'''
    generator = ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False)

    '''--------------------------------
    # 计算依赖于数据的变换所需要的统计信息(均值方差等),
    # 只有使用featurewise_center，featurewise_std_normalization或zca_whitening时需要此函数。
    -----------------------------------'''
    generator.fit(x_train)

    '''------------------------------------------------
    # fit_generator，数据分批加载，并可实现数据增强:
        generator：数据生成器，不必一次性将数据全部输入来训练
        steps_per_epoch, 生成器为死循环，不知道每个epoch何时结束，
            所以指定每个epoch多少个batch，一般的数值即为len(y)/batch_size
    # flow: 数据生成器接收numpy数组和标签为参数,
        生成经过数据提升或标准化后的batch数据，
        并在一个无限循环中不断的返回batch数据
    ----------------------------------------------------'''
    model.fit_generator(generator.flow(
                            x_train, y_train,
                            batch_size=batch_size),
                        steps_per_epoch=10,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

    '''----------------------
    train_on_batch，作用同fit, fit_generator
        适用于对Keras模型进行精细控制
        函数接受单批数据，执行反向传播，然后更新模型参数。
    --------------------------'''
    # model.fit(x_train, y_train)


# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
# 数据保存
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
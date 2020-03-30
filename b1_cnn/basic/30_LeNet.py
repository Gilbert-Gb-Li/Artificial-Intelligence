# coding: utf-8

"""
lenet5 模型
用于手写数字识别
"""

import numpy as np
from keras.layers import Input, Add, Conv2D, Dense, ZeroPadding2D, Activation, MaxPooling2D, Flatten
from keras.models import Model
from keras.utils import to_categorical

# 加载数据集。
# train.csv文件中共有42000个样本，将其分为40000个训练集和2000个验证集。
# test.csv文件中有42000个样本。
train = np.loadtxt('E:/dl_data/lenet5_data/train.csv', delimiter=',', skiprows=1)
data_train = train[:40000]
data_val = train[40000:]
data_test = np.loadtxt('E:/dl_data/lenet5_data/test.csv', delimiter=',', skiprows=1)

'''# 输出各个集合的形状'''
print(train.shape)
print(data_train.shape)
print(data_val.shape)
print(data_test.shape)


def lenet_5(input_shape=(32, 32, 1), classes=10):
    X_input = Input(input_shape)
    X = X_input
    X = ZeroPadding2D((2, 2))(X_input)  # 将输入调整为Paper内同样大小。
    X = Conv2D(6, (5, 5), strides=(1, 1), padding='valid', name='conv1')(X)
    X = Activation('tanh')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    X = Conv2D(16, (5, 5), strides=(1, 1), padding='valid', name='conv2')(X)
    X = Activation('tanh')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    X = Flatten()(X)
    X = Dense(120, activation='tanh', name='fc1')(X)
    X = Dense(84, activation='tanh', name='fc2')(X)
    X = Dense(10, activation='softmax')(X)
    model = Model(inputs=X_input, outputs=X, name='lenet_5')
    return model


model = lenet_5(input_shape=(28, 28, 1), classes=10)
model.summary()

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

X_train = data_train[:, 1:].reshape((data_train.shape[0], 28, 28, 1))
Y_train = to_categorical(data_train[:, 0])
model.fit(X_train, Y_train, epochs=10, batch_size=16)

X_val = data_val[:, 1:].reshape((data_val.shape[0], 28, 28, 1))
Y_val = to_categorical(data_val[:, 0])
preds = model.evaluate(X_val, Y_val)
print("Validation loss=" + str(preds[0]))
print("Validation accuracy=" + str(preds[1]))

X_test = data_test.reshape((data_test.shape[0], 28, 28, 1))
predicted = np.argmax(model.predict(X_test), axis=1)
with open('data/submission.csv', 'w') as f:
    f.write('ImageId,Label\n')
    for i in range(len(predicted)):
        f.write(str(i + 1) + ',' + str(predicted[i]) + '\n')
print(predicted)


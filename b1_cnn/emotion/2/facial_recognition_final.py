from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers.normalization import BatchNormalization


batch_size = 64
num_classes = 7
epochs = 30
img_rows, img_cols = 48, 48
#测试集数量
test_num  = 2000

import pandas as pd
import numpy as np
#data_train = pd.read_csv("G:/dl_data/testdata.csv")
data_train = pd.read_csv("G:/dl_data/fer2013.csv")
data_train.drop(['Usage'], axis=1, inplace=True)
data_train_pixels = data_train['pixels'].str.split(' ',-1,True);
data_train.drop(['pixels'], axis=1, inplace=True)
data_train = pd.concat([data_train, data_train_pixels], axis=1)
train_np = data_train.as_matrix()
#训练集、测试集
y_train = train_np[test_num:, 0]
X_train = train_np[test_num:, 1:]
y_test = train_np[:test_num, 0]
X_test = train_np[:test_num, 1:]
#图像预处理 归一化处理
X_train = X_train.astype('float32')
X_train = X_train - np.mean(X_train, 1, keepdims=True)
X_train = X_train / (np.std(X_train, 1, keepdims=True) + 1e-6)
X_test = X_test.astype('float32')
X_test = X_test - np.mean(X_test, 1, keepdims=True)
X_test = X_test / (np.std(X_test, 1, keepdims=True) + 1e-6)

X_train = X_train.reshape(X_train.shape[0], img_rows,img_cols,1)
X_test = X_test.reshape(X_test.shape[0], img_rows,img_cols,1)
input_shape = (img_rows, img_cols, 1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
#model.add(Conv2D(6, kernel_size=(5, 5),activation='relu',input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(5, 5),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(16, (5, 5), activation='relu'))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer="sgd",  #
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

'''
This example demonstrates the use of Convolution1D for text classification.
---------------------------------------------
Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.
'''

from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb

# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(path='D:/dl_data/imdb.npz', num_words=max_features)
print(x_train[0])
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
'''
# keras只能接受长度相同的序列输入，若序列长度不同需要使用pad_sequences()
# keras.preprocessing.sequence.pad_sequences(sequences, 
# 	maxlen=None,
# 	dtype='int32',
# 	padding='pre',
# 	truncating='pre', 
# 	value=0.)
# 参数
    sequences: 待padding列表 [n_samples, nb_time_steps]
    maxlen   : 设置最长时间步
    dtype    : 返回的数据类型
    padding  : 'pre'或'post'，补0时在序列的起始还是结尾补
    truncating: pre或post，截断序列时从起始还是结尾截断
    value    : 浮点数，此值将在填充时代替默认的填充值0
'''
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
print(x_train[0])
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions

''' embedding '''
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

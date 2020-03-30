import numpy as np
from keras.callbacks import TensorBoard
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
import _10_metrixs

"""==================
keras的基本用法
--------------
keras 实现全连接网络
keras for tensorflow
======================"""

# 构建数据集
X_data = np.array([1, 2, 3])
# X_data.shape： (3,)
# y_data = np.array([3, 1, 4])
X_data1 = np.array([[1], [2], [3]])
# "X_data1.shape": (3, 1)
X_data2 = np.array([[1, 1], [2, 1], [3, 1]])
# X_data2.shape: (3, 2)
X_data3 = np.array([[[1, 1]], [[2, 1]], [[3, 1]]])
# X_data3.shape: (3, 1, 2)

"""----------------------
构建神经网络
-------------------------"""
''' #
1.1 序列模型创建
可直接传入layer的list
[Dense(32, units=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax')]
# '''
model = Sequential()

""" #
1.2 使用add方法添加全连接层
Dense对象参数：
    units, 本层输出维度
    input_shape, 首层输入的维度，不包含batch_size，需与输入格式匹配
# """

model.add(Dense(12, input_shape=(2,), init='normal', activation='relu'))
# X_data3
# model.add(Dense(12, input_shape=(1, 2), init='normal', activation='relu'))

""" # 
X 维度与input_shape相关
    [batch_size, f1, f2...] => input_shape: (f1, f2...)
y 维度与input_shape及output相关 
    input_shape: (f1, f2...)
    output: n，最后一层Dense的输出个数
    y.shape: [batch_size, f1, f2..., n]
    因y_data维度会抛出以下异常
    # expected dense_2 to have 3 dimensions, but got array with shape (3, 1)
    解决：
        1. 将y_data指定为模型需要维度[None, 1, 2]
        2. 使用Flatten()展平为模型需要的维度
# """
# y_data = np.array([[[3, 2]], [[1, 4]], [[4, 1]]])
y_data = np.array([[3, 2], [1, 4], [4, 1]])
# model.add(Flatten())
model.add(Dense(2))
"""
2. compile主要完成损失函数和优化器的一些配置，初始化
参数：
    optimizer, 指定使用的优化器为SGD
        lr为学习率
    loss, loss函数
"""
sgd = SGD(lr=0.1)
model.compile(loss='mean_squared_error', optimizer=sgd)
"""
3. 训练 fit fit_generator train_on_batch
https://blog.csdn.net/xovee/article/details/91357143
参数：
    nb_epoch, 数据迭代的轮数
    batch_size, 每次数据迭代使用的批量
    verbose=1, 输出每epoch的误差及耗时
    callbacks, 回调函数，可添加评价指标等
    class_weight,
    sample_weight, 对个别样本加权
    initial_epoch,
    shuffle, 
    validation_data, 同时测试
"""

model.fit(X_data2, y_data,
          nb_epoch=10, batch_size=10, verbose=1,
          validation_data=(X_data2, y_data),
          callbacks=[TensorBoard(log_dir=""), _10_metrixs.Metrics()]
          )

"""4. 预测"""
y_predict = model.predict(X_data2)
print(y_predict)

"""# 输出模型各层的参数状况 #"""
model.summary()

'''# 输出每层信息'''
# for layer in model.get_layer(index=0):
# for layer in model.layers[:]:
#     print(layer.output)
    # print(layer.output_shape)
"""--------------------
5. 模型的保存与加载
-----------------------"""

''' ##
5.1. 模型保存
    .h5 表示以HDF5格式存储
    HDF5是一种存储相同类型数值的大数组的机制，
    适用于可被层次性组织且数据集需要被元数据标记的数据模型
'''
# model.save('regression.h5')

''' ##
5.2. 模型加载
## '''
# model = load_model('regression.h5')

''' ##
5.3. 只保存权重不保存模型
## '''
# model.save_weights('regression_weight.h5')
# model.load_weights('regression_weight.h5')

''' ##
5.4. 将模型保存为json
## '''
# json_string = model.to_json()
# model = model_from_json(json_string)


'''-----------
# 可视化
--------------'''
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.scatter(X_data, y_data)
# ax.plot(X_data, y_predict, 'r-', lw=5)
# plt.show()



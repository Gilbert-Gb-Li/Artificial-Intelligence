import tensorflow as tf

'''
tf 底层api
----------
input, 输入的形式[B, H, W, C]
filter, 卷积核输入形式[B, H, C_in, C_out]
strides, 步长[B, H, W, C]
'''
c_out = 128
'''# 通过输入的数据获取shape'''
b, h, w, c = input.get_shape()
'''
定义filter， 名为kernel；之后操作可以使用该名称提取变量
'''
filter = tf.get_variable('kernel', [3, 3, c, c_out])
tf.nn.conv2d(input,
             filter=filter,
             strides=[1, 2, 2, 1],
             padding='SAME',
             use_cudnn_on_gpu=False,  # 是否是gpu加速
             data_format='NHWC',  # NHWC == BHWC
             dilations=[1, 2, 2, 1],  # 空洞卷积，在卷积过程中补零不增加可训练参数的同时增加感受野
             name=None)  # 名字，用于tensorboard图形显示


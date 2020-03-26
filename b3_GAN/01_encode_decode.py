"""
编解码模型
--------
图像去噪与图像的超分辨率采样
"""
import tensorflow as tf

img_noise = tf.placeholder(tf.bfloat16, [32, 28, 28, 1])
# 无噪声图像
img_clear = tf.placeholder(tf.bfloat16, [32, 28, 28, 1])

'''1. 编码器'''
net = tf.layers.conv2d(img_noise, 32, 3, 1,
                       activtation=tf.nn.relu,
                       padding="same")
net = tf.layers.max_pooling2d(net, 2, 2)
net = tf.layers.conv2d(net, 64, 3, 1,
                       activtation=tf.nn.relu,
                       padding="same")
net = tf.layers.max_pooling2d(net, 2, 2)
# net.get_shape(): [32, 7, 7, 1]

'''2. 解码器'''
net = tf.layers.conv2d_transpose(net, 32, 3, strides=2, padding='same', activation=tf.nn.relu)
net = tf.layers.conv2d_transpose(net, 32, 3, strides=2, padding='same', activation=tf.nn.relu)
# net.get_shape(): [32, 28, 28, 1]
'''
# 若再添加反卷积操作，则图像分辨率为[32, 56, 56, 1]
# 此时可以完成超分辨率采样
net = tf.layers.conv2d_transpose(net, 32, 3, strides=2, padding='same', activation=tf.nn.relu)
'''
out_img = tf.layers.conv2d(net, 1, 3, strides=2, padding='same', activation=tf.nn.relu)

'''3. 损失函数'''
loss = tf.reduce_mean((out_img - img_clear) ** 2)

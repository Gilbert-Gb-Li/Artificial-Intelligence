"""
图片生成
-------
添加噪声，产生一定的随机性
"""
import tensorflow as tf

'''---------------
# 输入X：OneHot标签
#  噪声：使输出具有一定的随机性
# 输入y：对应的图像 
------------------'''
input_x_label = tf.placeholder(
    tf.bfloat16, [None, 10])
# 添加noise使输出具有一定的随机性
input_noize = tf.placeholder(
    tf.bfloat16, [None, 100])
# 标签为图像
input_y_image = tf.placeholder(
    tf.bfloat16, [None, 28, 28, 1])  
# 将OneHot变为7*7
'''-----------------
# 输入为X与噪声的合并
--------------------'''
inputs = tf.concat([input_x_label, input_noize], axis=-1)
net = tf.layers.dense(
    inputs, 49, activation=tf.nn.relu) 
net = tf.reshape(net, [-1, 7, 7, 1]) 
net = tf.layers.conv2d_transpose(
    net, 
    32, 
    3, 
    strides=2, 
    activation=tf.nn.relu, 
    padding="same"
)
net = tf.layers.conv2d_transpose(
    net, 
    64, 
    3, 
    strides=2, 
    activation=tf.nn.relu, 
    padding="same"
) 
out_image = tf.layers.conv2d(
    net, 1, 3, strides=1, padding="same")
loss = tf.reduce_mean((out_image - input_y_image) ** 2)
print(f"Net:{net.get_shape()}, Img:{out_image.get_shape()}")

with tf.Session() as sess:
    pass
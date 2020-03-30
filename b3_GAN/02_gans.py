import tensorflow as tf


def G(z, y):
    """
    由噪音与标签生成图像
    :param z: 噪声
    :param y: 标签
    :return:
    """
    with tf.variable_scope("GEN", reuse=tf.AUTO_REUSE):
        '''# 将噪声与标签直接合并'''
        inputs = tf.concat([z, y], axis=1)
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
    return out_image


def D(img, lab):
    """判别器，判别图像真伪"""
    with tf.variable_scope("DIS", reuse=tf.AUTO_REUSE):
        net = tf.layers.conv2d(img, 32, kernel_size=3, strides=1, activation=tf.nn.relu, padding="same")
        net = tf.layers.max_pooling2d(net, 2, 2)
        net = tf.layers.conv2d(net, 64, 3, strides=1, activation=tf.nn.relu, padding="same")
        net = tf.layers.max_pooling2d(net, 2, 2)
        image = tf.layers.flatten(net)
        '''----------------
        # 类似于逻辑回归
        # 相当于输入img与lab
        # 输出0,1值
        --------------------'''
        net = tf.concat([image, lab], axis=1)
        net = tf.layers.dense(net, 128, activation=tf.nn.relu)
        logit = tf.layers.dense(net, 2)
    return logit


'''# 设定输入格式 '''
input_label = tf.placeholder(
    tf.bfloat16, [32, 10])
input_noize = tf.placeholder(
    tf.bfloat16, [32, 100])
input_image = tf.placeholder(
    tf.bfloat16, [32, 28, 28, 1])

'''--------------------
# 生成器损失函数和构建过程
#   X： logit_out，即 D(G(z,y)
#   y： label，全1矩阵，即希望所有结果都准确
# 使用交叉熵构成loss
    优化函数只计算生成器的梯度，不必计算判别器的梯度
-----------------------'''
img_gen = G(input_noize, input_label)
logit_out = D(img_gen, input_label)
g_label = tf.one_hot(tf.ones([32]), 2)

loss_gen = ...

'''
# 判别器损失函数和构建过程
# 使用交叉熵构成loss
    优化函数只计算判别器的梯度，不必计算判别器的梯度
'''
img_gen = G(input_noize, input_label)
# 由生成器生成的图片判别结果
logit_fake = D(img_gen, input_label)
# 真实图片的判别结果
logit_true = D(input_image, input_label)
# logit_faked尽可能接近0，logit_true为1
label = tf.concat([tf.zeros([32]), tf.ones([32])])
oh_label = tf.one_hot(label, 2)

loss_dis = ...

'''
# 训练时，生成器与判别器交替训练
# 可以多几轮生成器训练后再进行1轮判别器训练
'''

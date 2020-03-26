import tensorflow as tf
import time

num_classes = 7
learning_rate = 3e-5

x = tf.placeholder(tf.float32, [None, 48, 48, 1])
y = tf.placeholder(tf.uint8, [None, num_classes])


def conv_block(net, num_iter, filters):
    """
    定义多个CNN组合单元
    """
    for itr in range(num_iter):
        net = tf.layers.conv2d(net, filters, 3, activation=tf.nn.relu, padding="same")
    return net


net = conv_block(x, 2, 64)
net = tf.layers.max_pooling2d(net, 2, 2)

net = conv_block(net, 1, 128)
net = conv_block(net, 1, 256)
net = tf.layers.max_pooling2d(net, 2, 2)

net = conv_block(net, 1, 512)
net = tf.layers.max_pooling2d(net, 2, 2)

net = tf.layers.flatten(net)
net = tf.layers.dense(net, 2048, activation=tf.nn.relu)
drop_out_rate3 = tf.placeholder(tf.float32)
net = tf.layers.dropout(net, drop_out_rate3)

net = tf.layers.dense(net, 2048, activation=tf.nn.relu)
logits = tf.layers.dense(net, 7, activation=None)
# 计算loss函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=y,
    logits=logits
))
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1)), tf.float16))
# 优化
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

from read_data import Data

epoches = 100
batch_size = 32
max_train_batches = 33886 // batch_size
test_batch_size = 32
max_test_batches = 2000 // test_batch_size
data = Data('data.npz')

with tf.Session() as sess:
    all_var = tf.global_variables()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # saver.restore(sess, 'model_saved/emotion')
    # print('读取模型成功')

    for i in range(epoches):
        start = time.time()
        # 训练loss
        total_loss = 0
        for j in range(max_train_batches):
            x_this_batch, y_this_batch = data.get_random_train_batch(batch_size)
            train_step.run(feed_dict={
                x: x_this_batch,
                y: y_this_batch,
                drop_out_rate3: 0.5
            })
            if j % 300 == 1:
                _loss = sess.run(loss, feed_dict={
                    x: x_this_batch,
                    y: y_this_batch,
                    drop_out_rate3: 1
                })
                total_loss += _loss
                print(f'{j - 1} / {max_train_batches} batch完成')

        # 测试集精度
        total_acc = 0
        for k in range(1, max_test_batches + 1):
            test_x, test_y = data.get_batch_test_data(test_batch_size)
            total_acc += sess.run(acc, feed_dict={
                x: test_x,
                y: test_y,
                drop_out_rate3: 1
            })
        end = time.time()
        print(
            f'1 epoch 完成，测试集精度{total_acc / max_test_batches},\ntotal_loss={total_loss / max_train_batches},\nepoch用时：{end - start}')
    with open('a.txt', 'a') as a:
        a.write(
            f'1 epoch 完成，测试集精度{total_acc / max_test_batches},\ntotal_loss={total_loss / max_train_batches},\nepoch用时：{end - start}')
    saver.save(sess, 'model_saved/emotion')
    print('模型保存完成')

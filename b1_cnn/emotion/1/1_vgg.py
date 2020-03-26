import time
import tensorflow as tf

num_classes = 7
learning_rate = 1e-4

x = tf.placeholder(tf.float32, [None, 48, 48, 1])
y = tf.placeholder(tf.uint8, [None, num_classes])
handled_x = tf.image.resize(x, [224, 224])


def block(net, num_iter, filters, blockID):
    """
    定义多个CNN组合单元
    """
    with tf.variable_scope("block%d" % blockID):
        for itr in range(num_iter):
            net = tf.layers.conv2d(net,
                                   filters, 3,
                                   activation=tf.nn.relu,
                                   padding="same")
        net = tf.layers.max_pooling2d(net, 2, 2)
    return net


net = block(handled_x, 2, 64, 1)
net = block(net, 2, 128, 2)
net = block(net, 2, 256, 3)
net = block(net, 2, 512, 4)
net = block(net, 2, 512, 5)
net = tf.layers.flatten(net)
net = tf.layers.dense(net, 2048, activation=tf.nn.relu)
drop_out_rate = tf.placeholder(tf.float32)
net = tf.layers.dropout(net, drop_out_rate)
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
batch_size = 10
max_train_batches = 33886 // batch_size
test_batch_size = 10
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
            })
            if j % 300 == 1:
                _loss = sess.run(loss, feed_dict={
                    x: x_this_batch,
                    y: y_this_batch,
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
            })
        end = time.time()
        print(
            f'1 epoch 完成，测试集精度{total_acc / max_test_batches},\ntotal_loss={total_loss / max_train_batches},\nepoch用时：{end - start}')
    with open('a.txt', 'a') as a:
        a.write(
            f'1 epoch 完成，测试集精度{total_acc / max_test_batches},\ntotal_loss={total_loss / max_train_batches},\nepoch用时：{end - start}')
    saver.save(sess, 'model_saved/emotion')
    print('模型保存完成')

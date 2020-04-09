import tensorflow as tf

"""================
# tf one hot
# 参数：
    indices     : 需要one-hot的标签, 可以为多维
    depth       : 输出向量的长度
    on_value    : indices[j] = i 时填充输出的值的标量，默认为1
    off_value   : indices[j] != i 时填充输出的值的标量，默认为0
==================="""
labels = [[0, 1], [1, 0]]
res = tf.one_hot(indices=labels, depth=4, on_value=1.0, off_value=0.0, axis=-1)
with tf.Session() as sess:
    print(sess.run(res))
# 输出：
# [[[ 1.  0.  0.  0.]
#   [ 0.  1.  0.  0.]]
#
#  [[ 0.  1.  0.  0.]
#   [ 1.  0.  0.  0.]]]
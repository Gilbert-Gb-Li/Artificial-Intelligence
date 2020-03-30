import tensorflow as tf
'''# Create a saver #'''
saver = tf.train.Saver()
'''# 获取程序中的变量 #'''
var_list = tf.global_variables()
sess = tf.Session()
for step in range(1000000):
    sess.run(tf.initialize_all_variables())
    if step % 1000 == 0:
        ''' 保存的是模型参数 '''
        saver.save(sess, 'my-model', global_step=step)

saver.restore(sess, tf.train.latest_checkpoint('./'))
saver.restore(sess, 'my-model-999000')
sess.close()
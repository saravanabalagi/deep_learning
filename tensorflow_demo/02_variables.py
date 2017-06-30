import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

node_1 = tf.placeholder(3.0, dtype=tf.float32)
node_2 = tf.constant(2.0)

node_3 = tf.add(node_1, node_2)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print('Running session')
	print(sess.run(node_3))


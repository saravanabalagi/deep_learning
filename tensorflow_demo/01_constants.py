import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

node_1 = tf.constant(1.0)
node_2 = tf.constant(2.0)

node_3 = tf.add(node_1, node_2)

with tf.Session() as sess:
	print('Running session')
	print(sess.run(node_3))


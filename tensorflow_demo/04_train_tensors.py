import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

w = tf.Variable([0.], tf.float32)
b = tf.Variable([0.], tf.float32)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

logit = w * x + b
loss = tf.reduce_sum(tf.square(logit - y))
optimizer = tf.train.GradientDescentOptimizer(0.003)
train = optimizer.minimize(loss)

input = {
	x:[4,5,6,7],
	y:[1,2,3,4]
}

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(1000):
		sess.run(train, input)
		curr_w, curr_b, curr_l = sess.run([w, b, loss], input)
	print(curr_w, curr_b, curr_l)
"""
vanilla neural networks
"""

import numpy as np
import tensorflow as tf
from utils import *
import time

RESTORE=True
batch_size = 10

x = tf.placeholder(shape=(None, 784), dtype=tf.float32)
y = tf.placeholder(shape=(None, 10), dtype = tf.int64)

m, v = tf.nn.moments(x, axes=1, keep_dims=True)
normalized_x = tf.div(x-m, tf.sqrt(v))

W1 = tf.Variable(initial_value=np.random.randn(784, 30)/np.sqrt(784), dtype=tf.float32)
b1 = tf.Variable(initial_value=np.zeros((30,), dtype=np.float32), dtype=tf.float32)

W2 = tf.Variable(initial_value=np.random.randn(30, 10)/np.sqrt(30), dtype=tf.float32)
b2 = tf.Variable(initial_value=np.zeros((10,), dtype=np.float32), dtype=tf.float32)

activation1 = tf.nn.sigmoid(tf.matmul(normalized_x, W1) + b1)
activation2 = tf.matmul(activation1, W2) + b2

loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=activation2)

match = tf.equal(tf.argmax(activation2, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_sum(tf.cast(match, tf.float32))/5

opt = tf.train.GradientDescentOptimizer(0.01)
train_op = opt.minimize(loss)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    saver.restore(sess, 'mymodel')
    print('Check point restored!')

    data = np.load('test.npy') # data.shape 5 x 28 x 28 x 3
    input_x = np.mean(data, axis=3).reshape((5, 784))
    I = np.eye(10, 10, dtype=float)
    input_y = np.zeros((5, 10), dtype=np.float32)
    input_y[0, :] = I[0, :]
    input_y[1, :] = I[1, :]
    input_y[2, :] = I[2, :]
    input_y[3, :] = I[4, :]
    input_y[4, :] = I[5, :]

    _accuracy, _logits = sess.run([accuracy, activation2], feed_dict={x: input_x, y: input_y})
    print(_accuracy)
    print(np.argmax(_logits, axis=1))

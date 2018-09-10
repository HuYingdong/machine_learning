"""
computation graph
"""

import tensorflow as tf
import numpy as np

a = np.arange(0, 9).reshape(3,3).astype(np.float32)
#print(a)
b = np.arange(6, 15).reshape(3,3).astype(np.float32)
#print(b)

# building computation graph
A = tf.convert_to_tensor(a)
B = tf.convert_to_tensor(b)
C = tf.matmul(a, b)
S = tf.sin(C)
D = tf.div(B, S)


#
with tf.Session() as sess:
    _d = sess.run(D)
#     _c, _s, _d = sess.run([C, S, D])
#
# print(type(_c))
# print(_c)
# print(_s)

print(_d)
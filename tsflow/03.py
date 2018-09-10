"""
convolutions
"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from scipy import misc

lena = misc.imread('lena.jpg')[:,:,1].reshape(1,512,512,1).astype(np.float32)

L = tf.convert_to_tensor(lena)
filter = np.ones((10,10,1,1), dtype=np.float32)/100
F = tf.convert_to_tensor(filter)

conv = tf.nn.conv2d(L, F, strides=[1,1,1,1], padding='VALID')

with tf.Session() as sess:
    _conv = sess.run(conv)
    print(_conv.shape)
    plt.imshow(np.squeeze(_conv), cmap='gray')
    plt.show()



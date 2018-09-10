"""
variables and gradients
"""
import tensorflow as tf

b = tf.Variable([2.7, 1.6])
c = tf.reduce_sum(tf.pow(b, 2))
d = - 3.0 * tf.div(1.0, 1.0 + c)


opt = tf.train.GradientDescentOptimizer(0.1)
grad = opt.compute_gradients(d)
train_op = opt.apply_gradients(grad)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        _b, _grad = sess.run([b, grad])
        print(_b)
        print(_grad)

    # for i in range(100):
    #     _, _d, _b = sess.run([train_op, d, b])
    #     print('Step: {0}  Objective: {1:.3f}'.format(i, _d))
    #     print('Variable: ', _b)


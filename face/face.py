import os
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import slim
# from utils import *
import time

"""
1, load, preprocess and save images as .npy file
2, split data into training set and test set
3, Build and train CNN
4, Analyse result
"""


def preprocess_images(folder):
    sub_folders = os.listdir(folder)  # folder 是文件所在目录，本机上为 '/Users/tianwen/Desktop/faces'
    # 返回一个列表，列表元素为子目录名
    num_classes = len(sub_folders)  # class的数量
    labels = []  # 标签列表
    images = []  # 图像列表
    for i in range(num_classes):
        current_folder = os.path.join(folder, sub_folders[i])  # 当前子目录的完整路径
        print('Now processing ', current_folder)
        dirs = os.listdir(current_folder)  # dirs为一个包含所有当前子目录下的文件名
        num_samples = len(dirs)
        for j in range(num_samples):
            labels.append(i)  # 将索引i存至标签列表
            filename = os.path.join(current_folder, dirs[j])  # 单个图像文件的完整路径
            image = misc.imread(filename, mode='L')  # 读取文件为灰度图像
            resized = misc.imresize(image, (48, 48))  # 缩放图像
            images.append(resized)  # 将缩放后的图像存放入图像列表
    features = np.array(images)  # 将图像列表转化为numpy数组
    print(features.shape)
    np.save('faces48', features)  # 保存图像numpy数组至硬盘
    labels = np.array(labels)  # 将标签列表转化为numpy数组
    print(labels.shape)
    np.save('labels', labels)  # 保存标签numpy数组至硬盘


# preprocess_images('faces')


def show_images(filename):
    images = np.load(filename)
    idx = np.random.randint(low=0, high=images.shape[0], size=16)
    for j in range(16):
        plt.subplot(4, 4, j + 1)
        plt.imshow(images[idx[j], :, :], cmap='gray')
    plt.show()


# show_images('faces48.npy')


def split_data(image_file, label_file):
    images = np.load(image_file)
    labels = np.load(label_file)
    assert (images.shape[0] == labels.shape[0])
    sample_size = images.shape[0]
    image_height = images.shape[1]
    image_width = images.shape[2]

    feature_size = image_height * image_width
    training_size = 40000
    test_size = sample_size - training_size

    data = np.hstack((np.reshape(images, (sample_size, feature_size)), np.reshape(labels, (sample_size, 1))))
    print(data.shape)
    np.random.shuffle(data)

    training_data = data[:training_size, :]
    test_data = data[training_size:, :]

    training_images = np.reshape(training_data[:, :feature_size], (training_size, image_height, image_width))
    training_labels = training_data[:, -1]

    test_images = np.reshape(test_data[:, :feature_size], (test_size, image_height, image_width))
    test_labels = test_data[:, -1]
    return training_images, training_labels, test_images, test_labels


def main():
    batch_size = 10
    num_classes = 122

    x = tf.placeholder(shape=(batch_size, 48, 48, 1), dtype=tf.float32)
    y = tf.placeholder(shape=batch_size, dtype=tf.int64)

    m, v = tf.nn.moments(x, axes=[0, 1, 2, 3], keep_dims=True)
    normalized_x = tf.div(x - m, tf.sqrt(v))

    conv1 = slim.conv2d(normalized_x, num_outputs=16, kernel_size=[3, 3], activation_fn=tf.nn.tanh)
    print('conv1 shape:', conv1.get_shape())

    pool1 = slim.max_pool2d(conv1, kernel_size=[2, 2])
    print('pool1 shape:', pool1.get_shape())  # 24 x 24 x 16

    conv2 = slim.conv2d(pool1, num_outputs=32, kernel_size=[3, 3], activation_fn=tf.nn.tanh)
    print('conv2 shape:', conv2.get_shape())

    pool2 = slim.max_pool2d(conv2, kernel_size=[2, 2])
    print('pool2 shape:', pool2.get_shape())  # 12 x 12 x 32

    conv3 = slim.conv2d(pool2, num_outputs=64, kernel_size=[3, 3], activation_fn=tf.nn.tanh)
    print('conv3 shape:', conv3.get_shape())

    pool3 = slim.max_pool2d(conv3, kernel_size=[2, 2])
    print('pool3 shape:', pool3.get_shape())  # 6 x 6 x 64

    flatten = slim.flatten(pool3)
    print('flatten shape:', flatten.get_shape())
    fc = slim.fully_connected(flatten, 64)
    logits = slim.fully_connected(fc, num_classes, activation_fn=None)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

    match = tf.equal(tf.argmax(logits, axis=1), y)
    accuracy = tf.reduce_sum(tf.cast(match, tf.float32)) / batch_size

    opt = tf.train.GradientDescentOptimizer(0.01)
    train_op = opt.minimize(loss)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        X_train, Y_train, X_test, Y_test = split_data('faces48.npy', 'labels.npy')
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
        n1 = X_train.shape[0] // batch_size
        n2 = X_test.shape[0] // batch_size

        sess.run(init)
        for i in range(10):
            lt = time.time()
            for j in range(n1):
                input_x = X_train[j * batch_size: (j + 1) * batch_size, ...]
                input_y = Y_train[j * batch_size: (j + 1) * batch_size]
                _, _loss = sess.run([train_op, loss], feed_dict={x: input_x, y: input_y})

                if (j + 1) % 100 == 0:
                    print('Epoch {0} step {1} loss {2:.3f}'.format(i + 1, j + 1, _loss))

            test_accuracy = np.zeros(n2)
            test_loss = np.zeros(n2)
            for j in range(n2):
                input_x = X_test[j * batch_size: (j + 1) * batch_size, ...]
                input_y = Y_test[j * batch_size: (j + 1) * batch_size]
                test_accuracy[j], test_loss[j] = sess.run([accuracy, loss],
                                                          feed_dict={x: input_x, y: input_y})
            print("=============================================================")
            print('Epoch {0}: ({3:.1f} sec) test loss {1:.3f}\t test accuracy {2:.2f}%' \
                  .format(i + 1, np.mean(test_loss), 100 * np.mean(test_accuracy), time.time() - lt))
            print("=============================================================")


# main()

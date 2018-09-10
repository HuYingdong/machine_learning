import numpy as np


def load_mnist():
    images = np.load('mnist_images.npy')
    sample_size = images.shape[0]
    feature_size = images.shape[1]*images.shape[2]
    images = images.reshape(sample_size, feature_size)
    labels = np.load('mnist_labels.npy')
    return images, labels


def create_blocks(images, labels, num_block):
    block_size = images.shape[0]//num_block
    image_blocks = []
    label_blocks = []
    for i in range(num_block):
        start_index = block_size * i
        end_index = block_size * (i + 1)
        _image_block = images[start_index:end_index, :]
        _label_block = labels[start_index:end_index]
        image_blocks.append(_image_block)
        label_blocks.append(_label_block)
        print('The {0}th block has components of size {1} and {2},indices range from {3} to {4}'
              .format(i, _image_block.shape, _label_block.shape, start_index, end_index))
    return image_blocks, label_blocks


def create_split(image_blocks, label_blocks, index):
    num_blocks = len(image_blocks)
    x_val = image_blocks[index]
    y_val = label_blocks[index]
    if index == 0:
        x_train = np.vstack(image_blocks[1:])
        y_train = np.hstack(label_blocks[1:])
    elif index == num_blocks - 1:
        x_train = np.vstack(image_blocks[:-2])
        y_train = np.hstack(label_blocks[:-2])
    else:
        x_train = np.vstack(image_blocks[:index] + image_blocks[index+1:])
        y_train = np.hstack(label_blocks[:index] + label_blocks[index+1:])
    return x_train, y_train, x_val, y_val


images, labels = load_mnist()

image_blocks, label_blocks = create_blocks(images, labels, 5)
x_train, y_train, x_val, y_val = create_split(image_blocks, label_blocks, 0)
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)


image_blocks = np.split(images, 5)
label_blocks = np.split(labels, 5)
x_train, y_train, x_val, y_val = create_split(image_blocks, label_blocks, 0)
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
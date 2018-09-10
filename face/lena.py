from matplotlib import pyplot as plt
from scipy import misc
import numpy as np

img = misc.imread('lena.jpg')
print(img.shape)
plt.imshow(img)


def conv(images, fliter):
    fs = fliter.shape[0]
    input_height, input_width = images.shape
    output_height = input_height - fs + 1
    output_width = input_width - fs + 1
    img_new = np.zeros((output_height, output_width))
    for i in range(output_height):
        for j in range(output_width):
            img_new[i, j] = np.sum(images[i:i + fs, j:j + fs] * fliter)
    return img_new


def sharpen(images, c):
    f = -np.ones((5, 5)) * c / 25
    f[2, 2] = 1 - np.sum(f)
    print(f)
    return conv(images, f)


output = conv(img[:, :, 0], np.ones((3, 3)) / 9)
sharpen_output = sharpen(img[:, :, 0], 4)

plt.subplot(1, 2, 1)
plt.imshow(img[:, :, 0], cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(output, cmap='gray')

f1 = np.ones((5, 5)) / 25
f2 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

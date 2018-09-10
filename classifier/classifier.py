
import time
import numpy as np
# from sklearn import datasets  # 内置数据集
from sklearn.neighbors import KNeighborsClassifier  # K最近邻(kNN，k-Nearest Neighbor)
from sklearn.linear_model import LogisticRegression  # logistic 回归
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # 线性判别分析
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis  # 二次判别分析
from sklearn.tree import DecisionTreeClassifier  # 决策树
from sklearn.svm import SVC  # 支持向量机
from sklearn.ensemble import RandomForestClassifier  # 随机森林
import matplotlib.pyplot as plt

# iris = datasets.load_iris()
# X = iris.data
# Y = iris.target
# Y = np.expand_dims(Y, 1)
# data = np.hstack((Y, X))
# np.random.seed(123)
# np.random.shuffle(data)
# X = data[:, 1:]
# Y = data[:, 0].astype(int)
# Y = np.squeeze(Y)
# X_train = X[:120, :]
# X_test = X[120:, :]
# Y_train = Y[:120]
# Y_test = Y[120:]

# digits = datasets.load_digits()
# X = digits.data
# Y = digits.target
# X_train = X[0:1200, :]
# X_test = X[1200:, :]
# Y_train = Y[0:1200]
# Y_test = Y[1200:]

images = np.load('mnist_images.npy')
print(images.shape)
labels = np.load('mnist_labels.npy')
print(labels.shape)

# 数据压缩
X = images.reshape((60000, 28*28))
X_train = X[:50000, :]
X_test = X[5000:, :]
Y_train = labels[:50000]
Y_test = labels[50000:]


def mypca(x, n):
    x = x.astype(np.double)
    assert(n < x.shape[1])
    print('Input array shape: ', x.shape)
    print('Number of principal components: {0}'.format(n))
    cov = np.dot(x.transpose(), x)/x.shape[0]
    w, v = np.linalg.eigh(cov)
    start_index = x.shape[1] - n
    sum_of_eigenvalues = np.sum(w)
    wn = w[start_index:]
    sum_of_principal_eigenvalues = np.sum(wn)
    percentage = sum_of_principal_eigenvalues/sum_of_eigenvalues
    print('Percentage of variance retained: {0:.1f}%'.format(100*percentage))
    eig_v = v[:, start_index:]
    output = np.dot(x, eig_v)
    print('Output array shape: ', output.shape)
    return output, eig_v


X_train_pca, eig_v = mypca(X_train, 40)
X_test_pca = np.dot(X_test, eig_v)


def shrink_images(images):
    assert [images.ndim == 3]
    assert [images.shape[1] % 2 == 0]
    assert [images.shape[2] % 2 == 0]
    subaray1 = images[:, ::2, ::2]  # 奇数行、奇数列
    subaray2 = images[:, 1::2, ::2]  # 偶数行、奇数列
    subaray3 = images[:, ::2, 1::2]  # 奇数行、偶数列
    subaray4 = images[:, 1::2, 1::2]  # 偶数行，偶数列
    avg = (subaray1 + subaray2 + subaray3 + subaray4)/4
    return avg.astype(np.uint8)


shrinked_images = shrink_images(images)
print(shrinked_images.shape)


# 作图
def show_images_and_labels(images, labels, offset=0):
    print(labels[offset:offset+64])
    plt.gray()
    for i in range(8):
        for j in range(8):
            plt.subplot(8, 8, 8 * i + j + 1)
            plt.imshow(images[offset + 8 * i + j, :, :])
    plt.show()


show_images_and_labels(images, labels, offset=100)
show_images_and_labels(shrinked_images, labels, offset=100)


X = shrinked_images.reshape(60000, 196)
X_train = X[:50000, :]
X_test = X[50000:, :]
Y_train = labels[:50000]
Y_test = labels[50000:]


classifier = KNeighborsClassifier(n_neighbors=5)
classifier = LogisticRegression()
classifier = LinearDiscriminantAnalysis()
classifier = QuadraticDiscriminantAnalysis()
classifier = DecisionTreeClassifier(max_depth=20, min_samples_split=5, random_state=123, max_features=None)
classifier = SVC(kernel='linear')
classifier = RandomForestClassifier(max_depth=20, random_state=123)


start_time = time.time()
print('start trainging...')
classifier.fit(X_train, Y_train)
prediction = classifier.predict(X_test)
match = sum(prediction == Y_test)
print('Classification rate:{0}/{1} ({2:.1f}%)'.format(match, Y_test.shape[0], match/Y_test.shape[0]*100))
print('useing time:{0:.1f} sec'.format(time.time()-start_time))


import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier  # K最近邻(KNN，k-Nearest Neighbor)
from sklearn.linear_model import LogisticRegression  # logistic 回归
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # 线性判别分析
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis  # 二次判别分析
from sklearn.tree import DecisionTreeClassifier  # 决策树
from sklearn.svm import SVC  # 支持向量机
from sklearn.ensemble import RandomForestClassifier  # 随机森林


def load_mnist():
    images = np.load('mnist_images.npy')
    sample_size = images.shape[0]
    feature_size = images.shape[1] * images.shape[2]
    images = images.reshape(sample_size, feature_size)
    labels = np.load('mnist_labels.npy')
    return images, labels


images, labels = load_mnist()
data = np.hstack((images, labels.reshape(labels.shape[0], 1)))

np.random.seed(123)
np.random.shuffle(data)

X = data[:, :784]
X_train = X[:3000, :]
X_test = X[3000:, :]

Y = data[:, -1]
Y_train = Y[:3000]
Y_test = Y[3000:]


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


def training(x_train, x_test, y_train, y_test):
    start_time = time.time()
    print('start training...')
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    match = sum(prediction == y_test)
    print('Classification rate:{0}/{1} ({2:.1f}%)'.format(match, y_test.shape[0], match / y_test.shape[0] * 100))
    print('useing time:{0:.1f} sec'.format(time.time() - start_time))


# # KNN
# classifier = KNeighborsClassifier(n_neighbors=3)
# training(X_train, X_test, Y_train, Y_test)


# # Logistic Regression
# classifier = LogisticRegression()
# training(X_train, X_test, Y_train, Y_test)


# # LDA
# classifier = LinearDiscriminantAnalysis()
# training(X_train_pca, X_test_pca, Y_train, Y_test)


# # QDA
# classifier = QuadraticDiscriminantAnalysis()
# training(X_train_pca, X_test_pca, Y_train, Y_test)


# # Decision Tree
# classifier = DecisionTreeClassifier(max_depth=30, min_samples_split=5, random_state=123)
# training(X_train, X_test, Y_train, Y_test)


# # SVM
# classifier = SVC(kernel='poly')
# training(X_train, X_test, Y_train, Y_test)


# Random Forest
classifier = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=123)
training(X_train, X_test, Y_train, Y_test)





# 线性判别分析

from sklearn import datasets
import numpy as np
from scipy.stats import multivariate_normal

iris = datasets.load_iris()

X = iris.data
Y = iris.target

Y = np.expand_dims(Y, 1)

data = np.hstack((Y, X))

np.random.seed(123)
np.random.shuffle(data)

X = data[:, 1:]
Y = data[:, 0].astype(int)
Y = np.squeeze(Y)

X_train = X[:120, :]
X_test = X[120:, :]
Y_train = Y[:120]
Y_test = Y[120:]


class Linda:
    def __init__(self):
        self.num_classes = None  # 类数
        self.pi = None  # 先验概率
        self.mu = None  # 分类均值
        self.var = None  # 方差
        self.result = None  # 预测结果
        self.proba = None  # 概率

    def fit(self, X, Y):
        self.num_classes = np.max(Y) + 1

        self.pi = [None] * self.num_classes
        class_num = np.unique(Y, return_counts=True)[1]
        for i in range(self.num_classes):
            self.pi[i] = class_num[i] / len(Y)

        self.mu = [None] * self.num_classes
        for i in range(self.num_classes):
            self.mu[i] = X[Y == i, :].mean(axis=0)

        self.var = np.zeros(shape=(X.shape[1], X.shape[1]))
        diff = [None] * self.num_classes
        for i in range(self.num_classes):
            diff[i] = X[Y == i, :] - self.mu[i]
            self.var += (np.dot(diff[i].T, diff[i])) / (X.shape[0] - self.num_classes)

    def predict(self, X):
        self.result = [None] * X.shape[0]
        socre = [None] * self.num_classes
        for i in range(X.shape[0]):
            for j in range(self.num_classes):
                a = np.linalg.inv(self.var).dot(self.mu[j])
                socre[j] = X[i, :].T.dot(a) - 0.5 * self.mu[j].T.dot(a) + np.log(self.pi[j])
            self.result[i] = np.argmax(socre)
            self.result = np.array(self.result)

    def predict_proba(self, X):
        self.proba = np.zeros(shape=(X.shape[0], self.num_classes))
        numerator = [None] * self.num_classes
        prob = [None] * self.num_classes
        for i in range(X.shape[0]):
            for j in range(self.num_classes):
                mulnorm_pdf = multivariate_normal.pdf(X[i, :], mean=self.mu[j], cov=self.var)
                numerator[j] = self.pi[j] * mulnorm_pdf
            for j in range(self.num_classes):
                prob[j] = numerator[j] / sum(numerator)
            for j in range(self.num_classes):
                self.proba[i, j] = prob[j]


if __name__ == "__main__":
    lda = Linda()
    lda.fit(X_train, Y_train)
    print(lda.pi)
    print(lda.mu)
    print(lda.var)
    lda.predict(X_test)
    print(lda.result)
    print(Y_test)
    lda.predict_proba(X_test)
    print(lda.proba)

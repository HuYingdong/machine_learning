# 二次判别分析

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


class Quada:
    def __init__(self):
        self.num_classes = None  # 类数
        self.pi = None  # 先验概率
        self.mu = None  # 分类均值
        self.var = None  # 方差
        self.class_var = None  # 分类方差
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

        self.var = np.zeros((X.shape[1], X.shape[1]))
        self.class_var = [None] * self.num_classes

        diff = [None] * self.num_classes
        for i in range(self.num_classes):
            diff[i] = X[Y == i, :] - self.mu[i]
            self.class_var[i] = np.dot(diff[i].T, diff[i]) / (class_num[i] - 1)
            self.var += np.dot(diff[i].T, diff[i])
        self.var = self.var / (X.shape[0] - self.num_classes)

    def predict(self, X):
        self.predict_proba(X)
        self.result = [None] * X.shape[0]
        for i in range(X.shape[0]):
            self.result[i] = np.argmax(self.proba[i])
        self.result = np.array(self.result)

        # a = [None] * self.num_classes
        # b = [None] * self.num_classes
        # c = [None] * self.num_classes
        # for i in range(X.shape[0]):
        #     for j in range(self.num_classes):
        #         diff = X[i, :] - self.mu[j]
        #         a[j] = -0.5 * diff.T.dot(np.linalg.inv(self.var)).dot(diff)
        #         b[j] = -0.5 * np.log(np.linalg.det(self.class_var[j])) + np.log(self.pi[j])
        #         c[j] = a[j] + b[j]
        #     self.result[i] = np.argmax(c)
        #     self.result = np.array(self.result)

    def predict_proba(self, X):
        self.proba = np.zeros((X.shape[0], self.num_classes))
        numerator = [None] * self.num_classes
        prob = [None] * self.num_classes
        for i in range(X.shape[0]):
            for j in range(self.num_classes):
                mulnorm_pdf = multivariate_normal.pdf(X[i, :], self.mu[j], self.class_var[j])
                numerator[j] = self.pi[j] * mulnorm_pdf
            for j in range(self.num_classes):
                prob[j] = numerator[j] / sum(numerator)
            for j in range(self.num_classes):
                self.proba[i, j] = round(prob[j], 3)


if __name__ == "__main__":
    qda = Quada()
    qda.fit(X_train, Y_train)
    print(qda.pi)
    print(qda.mu)
    print(qda.var)
    print(qda.class_var)
    qda.predict(X_test)
    print(qda.result)
    print(Y_test)
    qda.predict_proba(X_test)
    print(qda.proba)

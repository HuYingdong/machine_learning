
from sklearn import datasets, linear_model
import numpy as np
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()

x = diabetes.data
print(x.shape)

y = diabetes.target
print(y.shape)

x = np.hstack((np.ones((x.shape[0], 1)), x))
x_train = x[:300, :]
x_test = x[300:, :]
y_train = y[:300]
y_test = y[300:]

# clf = linear_model.Ridge(alpha=0.1)
# clf = linear_model.Lasso(alpha=0.1)
clf = linear_model.LinearRegression()

clf.fit(x_train, y_train)
print(clf.coef_)
print(clf.intercept_)
beta = clf.coef_
beta[0] = clf.intercept_

prediction = x_test.dot(beta)
print(mean_squared_error(y_test, prediction))


# 类实现
class Linreg:
    def __int__(self):
        self.beta = None
        self.prediction = None
        self.mse = None

    def fit(self, x, y):
        a = np.linalg.inv(np.transpose(x).dot(x))
        b = np.transpose(x).dot(y)
        self.beta = a.dot(b)
        return self.beta

    def predict(self, x):
        self.prediction = x.dot(self.beta)
        return self.prediction

    def error(self, x, y):
        diff = y - self.prediction
        self.mse = np.linalg.norm(diff) ** 2 / x.shape[0]
        return self.mse
        # return (np.transpose(diff).dot(diff))/x.shape[0]

reg = Linreg()
reg.fit(x_train, y_train)
print(reg.beta)
reg.predict(x_test)
print(reg.prediction)
reg.error(x_test, y_test)
print(reg.mse)




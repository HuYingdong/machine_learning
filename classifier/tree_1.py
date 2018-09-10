import numpy as np


class Tree(object):
    def __init__(self):
        self.impurity = None
        self.cut_point = None
        self.col_index = None
        self.outcome = None

        self.left_child_tree = None
        self.right_child_tree = None

    @property
    def is_terminal(self):
        return not bool(self.left_child_tree and self.right_child_tree)


    def _getEntropy(self, y):
        """
        计算熵
        """
        num = y.shape[0]
        y_class_info = np.unique(y, return_counts=True)
        prob = y_class_info[1] / num
        entropy = np.sum(prob * np.log(prob))
        return -entropy


    def _get_split_value(self, x):
        """
        给定特征 x,返回对x 做分割的点
        :param x: 
        :return: 分割点构成的集合
        """
        split_value = set()
        x_unique = np.unique(x)
        for i in range(1, len(x_unique)):
            split_value.add((x_unique[i-1] + x_unique[i]) / 2)      # 将俩俩数据的平均数作为分割点
        return split_value

    def _get_split_data(self, x, y, column, value, return_x=False):
        """
        给定分割点，得到分割后的数据集
        """
        left_bool = (x[:, column] <= value)
        right_bool = (x[:, column] > value)
        if return_x:
            return x[left_bool], x[right_bool], y[left_bool], y[right_bool]
        else:
            return y[left_bool], y[right_bool]

    def _information_gain(self, y_left, y_right):
        """计算信息增益"""
        entropy = self._getEntropy(np.append(y_left, y_right))
        num = len(y_left) + len(y_right)
        information = self._getEntropy(y_left)*len(y_left) + self._getEntropy(y_right)*len(y_right)
        return entropy - information / num

    def _find_best_split(self, x, y):
        col_max, value_max, gain_max = None, None, 0.0
        for column in range(x.shape[1]):
            split_value = self._get_split_value(x[:, column])
            for value in split_value:
                y_left, y_right = self._get_split_data(x, y, column, value)
                information_gain = self._information_gain(y_left, y_right)
                if information_gain > gain_max:
                    col_max, value_max, gain_max = column, value, information_gain
        return col_max, value_max, gain_max

    def fit(self, X, Y, minimum_gain=0.001):
        try:
            column, value, gain = self._find_best_split(X, Y)
            assert (gain > minimum_gain)

            self.col_index = column
            self.cut_point = value
            self.impurity = gain

            left_X, right_X, left_Y, right_Y = self._get_split_data(X, Y, column, value, return_x=True)

            self.left_child_tree = Tree()
            self.left_child_tree.fit(left_X, left_Y, minimum_gain)

            self.right_child_tree = Tree()
            self.right_child_tree.fit(right_X, right_Y, minimum_gain)
        except AssertionError:
            self._get_leaf_value(Y)

    def _get_leaf_value(self, Y):
        """在叶上找到最优值"""
        y_unique, counts = np.unique(Y, return_counts=True)
        self.outcome = y_unique[np.argmax(counts)]

    def predict_row(self, row):
        """对单行做预测"""
        if not self.is_terminal:
            if row[self.col_index] < self.cut_point:
                return self.left_child_tree.predict_row(row)
            else:
                return self.right_child_tree.predict_row(row)
        return self.outcome

    def predict(self, X):
        result = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            result[i] = self.predict_row(X[i, :])
        return result.astype(int)

if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.tree import DecisionTreeClassifier

    iris = datasets.load_iris()

    X = iris.data
    Y = iris.target
    Y = np.expand_dims(Y, 1)
    # print(X.shape, Y.shape)
    data = np.hstack((Y, X))

    np.random.seed(123)
    np.random.shuffle(data)  # 将列表中的元素打乱

    X = data[:, 1:]
    Y = data[:, 0].astype(int)
    Y = np.squeeze(Y)

    X_train = X[0:120, :]
    Y_train = Y[0:120]
    X_test = X[120:, :]
    Y_test = Y[120:]
    tree = Tree()
    tree.fit(X_train, Y_train)
    print('由自编决策树得到的结论：', tree.predict(X_test), sep='\n')

    clf = DecisionTreeClassifier()
    clf.fit(X_train, Y_train)
    print('由sklearn的决策树得到的结论：', clf.predict(X_test), sep='\n')

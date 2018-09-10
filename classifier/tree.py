# 决策树 CART

import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
Y = iris.target
labels = []
for i in Y:
    if i == 0:
        labels.append(iris.target_names[0])
    elif i == 1:
        labels.append(iris.target_names[1])
    else:
        labels.append(iris.target_names[2])

labels = np.expand_dims(labels, 1)
data = np.hstack((X, labels))
np.random.seed(123)
np.random.shuffle(data)
data_train = data[:120, :]
data_test = data[120:, :]
feat_names = iris.feature_names

class CART:
    def __init__(self):
        pass

    # 计算数据集data的基尼值
    def calcGini(self, data):
        num_samples = data.shape[0]
        num_label = {}
        for each_row in data:  # 遍历dataSet中的每一行
            labels = each_row[-1]
            if labels not in num_label.keys():
                num_label[i] = 0
            num_label[i] += 1
        prob = 0.0
        for key in num_label:
            prob += num_label[key]/num_samples
        Gini = 1-(prob**2)
        return Gini

    # 使用二分法对data样本按第num个属性划分。对连续值，先排序，再按相邻点的中间值划分，共有n-1个划分点
    def split_value(self, data, num):
        split_value = []  # 划分点集合
        order_data = data[np.argsort(data[:, num])]  # 按第num列对行排序
        for i in range(len(data) - 1):  # n-1个划分点
            split_value.append((float(order_data[i, 2])+float(order_data[i + 1, 2]))/2)
        return split_value

    # 对data样本按第num个属性的value值二分样本
    def split_data(self, data, num, value):
        data_left = []  # 左边的样本
        data_right = []  # 右边的样本
        for each_row in data:  # 遍历数据所有行进行划分
            if float(each_row[num]) <= value:
                select_row = list(each_row[:num])  # 从第0个位置开始截取到num位置
                select_row.extend(list(each_row[num+1:]))  # 再从第num+1的位置截取该行，添加到之前的属性后面
                data_left.append(select_row)
            else:
                select_row = list(each_row[:num])  # 从第0个位置开始截取到num位置
                select_row.extend(list(each_row[num + 1:]))  # 再从第num+1的位置截取该行，添加到之前的属性后面
                data_right.append(select_row)
        return np.array(data_left), np.array(data_right)

        # 选出第num个属性的最优划分点及其基尼指数
    def calcGini_index(self, data, num):
        Gini_index = {}
        for value in self.split_value(data, num):
            data_left, data_right = self.split_data(data, num, value)
            Gini_index[value] = (len(data_left) * self.calcGini(data_left)+len(data_right) * self.calcGini(data_right))/len(data)
        min_Gini_index = min(Gini_index.values())
        best_value = float(list(Gini_index.keys())[list(Gini_index.values()).index(min_Gini_index)])
        return min_Gini_index, best_value

    # 在所有属性中选择最优划分属性
    def best_split(self, data):
        num_features = data.shape[1] - 1  # 减去最后一列的类标签，得到属性数
        Gini_index = {}
        for num in range(num_features):  # 遍历所有属性，得到各属性的基尼指数，求最小得出最优划分属性
            Gini_index[num] = self.calcGini_index(data, num)
        best_feature = int(min(Gini_index.items())[0])
        return best_feature

    def majorityCnt(self, classList):  # 少数服从多数
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1  # 该类标签下数据个数+1
        return int(max(classCount.items())[0])   # 返回数据表最大的那个类标签

    def createTree(self, data, feat_names):  # 主函数
        classList = list(data[:, -1])  # 保存类标签数组
        if classList.count(classList[0]) == len(classList):  # 如果classList所有数据同属一类
            return classList[0]  # 停止划分
        bestFeat = self.best_split(data)  # 找到最优的属性
        bestFeatName = feat_names[bestFeat]  # 找到该属性名
        myTree = {bestFeatName: {}}  # 以多级字典的形式展示树，类似多层json结构
        del (feat_names[bestFeat])  # 在feat_names数组中删除用来划分的属性
        bestValue = self.calcGini_index(data, bestFeat)[1]
        split_data = self.split_data(data, bestFeat, bestValue)
        leaf = ['<'+str(bestValue), '>'+'bestValue']
        for data in split_data:
            for leaf in leaf:
                subfeatnames = feat_names[:]  # 拷贝数组feat_names
                myTree[bestFeatName][leaf] = self.createTree(data, subfeatnames)  # 循环craeteTree函数
        return myTree  # 返回树


Cart = CART()
print(Cart.createTree(data_train, feat_names))

from math import log
import operator


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]  # 创建数据集
    labels = ['no surfacing', 'flippers']  # 分类属性
    return dataSet, labels


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  # 求样本矩阵的长度
    labelCounts = {}
    for featVec in dataSet:  # 遍历dataSet中的每一行
        currentLabel = featVec[-1]  # 每行中最后一个数表示数据的类标签
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0  # 如果标签数据中没有该标签，则添加该标签
        labelCounts[currentLabel] += 1  # 如果有，则标签数加1
    shannonEnt = 0.0
    for key in labelCounts:  # 遍历标签数组
        prob = float(labelCounts[key]) / numEntries  # 标签数除以样本矩阵的长度，即概率
        shannonEnt -= prob * log(prob, 2)  # 以2为底求对数
    return shannonEnt


'''
计算所有属性值得信息增益，并计算最佳划分方法
传入的参数，dataSet是样本矩阵，axis是第axis的位置，value是该位置的值
例：tree.splitDataSet(dataSet, 1, 1)：求dataSet中第2个位置是1的数组
'''

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:  # 遍历样本矩阵
        if featVec[axis] == value:  # 如果该行中axis位置的值为value
            reducedFeatVec = featVec[:axis]  # 从第0个位置开始截取到axis位置
            reducedFeatVec.extend(featVec[axis + 1:])  # 再从第axis+1的位置截取该行，添加属性到之前的属性后面
            retDataSet.append(reducedFeatVec)  # 一个个的添加reducedFeatVec数组
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 因为最后一行是存的类标签，这里是属性数
    baseEntropy = calcShannonEnt(dataSet)  # 原始熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 遍历除了类标签的样本矩阵
        featList = [example[i] for example in dataSet]  # example[0]=[1,1,1,0,0],example[1]=[1,1,0,1,1]
        uniqueVals = set(featList)  # set集合有去重作用，所以uniqueVals=[0,1]
        newEntropy = 0.0
        for value in uniqueVals:  # 遍历uniqueVals数组
            subDataSet = splitDataSet(dataSet, i, value)  # 划分数据集
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)  # 计算划分后的熵
        infoGain = baseEntropy - newEntropy  # 计算信息增益
        if (infoGain > bestInfoGain):  # 比较当前最大的信息增益
            bestInfoGain = infoGain  # 始终选择最大值
            bestFeature = i  # 返回最大值下的划分属性
    return bestFeature  # 返回信息增益最大的划分属性


'''
我们已经做到了寻找划分数据集的最佳属性，接下来就是递归调用，
不断划分下去。划分到什么时候结束呢？
这里有两个依据，第一，划分后的某个数据集中所有数据都同属于一类，这个时候就没必要再划分了，
再者，由于这里所讲的决策树是消耗属性的，所以当所有属性都用完了，划分也就停止了。
如果所有属性都用完了，某堆数据集中的数据仍不统一，解决方法就是少数服从多数
'''

def majorityCnt(classList):  # 少数服从多数
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1  # 该类标签下数据个数+1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  # 对标签数据个数降序排序
    return sortedClassCount[0][0]  # 返回数据表最大的那个类标签


def createTree(dataSet, labels):  # 相当于主函数
    classList = [example[-1] for example in dataSet]  # 保存类标签数组，classList=[‘yes’,’yes’,’no’,’no’,’no’]
    if classList.count(classList[0]) == len(classList):  # 如果classList全为‘yes’,说明所有数据同属一类
        return classList[0]  # 所以停止划分
    if len(dataSet[0]) == 1:  # 如果dataSet第一维的长度为1
        return majorityCnt(classList)  # 执行少数服从多数程序
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 找到信息增益最大的属性
    bestFeatLabel = labels[bestFeat]  # 找到该属性的类标签
    myTree = {bestFeatLabel: {}}  # 以多级字典的形式展示树，类似多层json结构
    del (labels[bestFeat])  # 在labels数组中删除用来划分的类标签
    featValues = [example[bestFeat] for example in dataSet]  # 把dataSet矩阵中属于用来划分的类标签的属性保存咋featValues数组中
    uniqueVals = set(featValues)  # 去掉数组中重复的值
    for value in uniqueVals:
        subLabels = labels[:]  # 拷贝数组labels,使其不会丢失掉它的属性
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)  # 循环craeteTree函数
    return myTree  # 返回树

a ,b = createDataSet()
c = createTree(a, b)
print(c)

def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


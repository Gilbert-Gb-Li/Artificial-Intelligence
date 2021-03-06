from math import log
import operator


def calcShannonEnt(dataSet):  # 计算数据的熵(entropy)
    """
    from math import log
        log(1/px) = -log(px)
        print(-(1/2) * log(1/2, 2) * 2)
        print(-(1/4) * log(1/4, 2) * 4)
        print(-(1/8) * log(1/8, 2) * 8)
    :param dataSet:
    :return:
    """
    numEntries = len(dataSet)  # 数据条数
    labelCounts = {}
    '''统计有多少个类以及每个类的数量'''
    for featVec in dataSet:
        currentLabel = featVec[-1]  # 每行数据的最后一个字（类别）
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 计算单个类的熵值
        shannonEnt -= prob * log(prob, 2)  # 累加每个类的熵值
    return shannonEnt


def createDataSet1():  # 创造示例数据
    dataSet = [['黑', '是', '男'],
               ['白', '否', '男'],
               ['黑', '是', '男'],
               ['白', '否', '女'],
               ['黑', '是', '女'],
               ['黑', '否', '女'],
               ['白', '否', '女'],
               ['白', '否', '女']]
    labels = ['头发', '近视']  # 两个特征
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    """按某个特征分类后的数据， 例如value = 长， axis为头发axis = 0"""
    retDataSet = []  # retDataSet返回的是数据中，所有头发为长的记录
    for featVec in dataSet:
        if featVec[axis] == value:
            # 每次会将这个feature删除掉
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):  # 选择最优的分类特征
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)  # 原始的熵
    bestInfoGain = 0
    bestFeature = -1
    '''（1）遍历每个特征'''
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0
        '''（2）对单个特征中每个唯一值都进行子树的切分，然后计算这个的信息增益，然后累加每个分裂条件的信息增益，为newEntropy'''
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)  # 按特征分类后的熵
        infoGain = baseEntropy - newEntropy  # 原始熵与按特征分类后的熵的差值
        if (infoGain > bestInfoGain):  # 若按某特征划分后，熵值减少的最大，则次特征为最优分类特征
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


'''# 按分类后类别数量排序，比如：最后分类为2男1女，则判定为男；'''


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


'''决策树核心逻辑： 递归，深度优先遍历创建子树'''


def createTree(dataSet, labels):
    # print(dataSet)
    classList = [example[-1] for example in dataSet]  # 类别：男或女
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:  # 当只剩最优一个feature时，因为每次分裂会删除一个特征
        return majorityCnt(classList)
    '''(1) 选择最优分裂特征'''
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择最优特征
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}  # 分类结果以字典形式保存
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    '''(2) 遍历分裂特征的每个唯一值，分裂产生子树'''
    for value in uniqueVals:
        # 将每个label都往子树传递，labels一直可能会被选用
        subLabels = labels[:]
        """
        # 对基于best feature这个feature内每个唯一值切分为不同的子树，然后返回子树的结果。每个子树切分能多个分支
        # 递归调用，每层的切分为每个特征的唯一值
        (3)对最佳分裂特征的每个值，分别创建子树
        """
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


if __name__ == '__main__':
    dataSet, labels = createDataSet1()  # 创造示列数据
    print(createTree(dataSet, labels))  # 输出决策树模型结果

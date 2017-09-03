# coding=utf-8
# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
from math import log
import operator


def classify(inputTree, featureLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featureIndex = featureLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featureIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featureLabels, testVec)
            else:
                classLabel = secondDict[key]

    return classLabel


'''
输入：
dataSet: 数据集，二维表形式（嵌套列表），数据集中的最后一项是包含分类的类标签
labels: 标签集，即每个属性对应的标签名列表
算法步骤：
1、从数据集中获取全部的分类的类别列表
2、设定递归退出条件：1）所有分类标签都属于同一类别时退出，此时表明已经正确分类  2) 当处理完所有特征，仍然存在未分类的样本时，则返回剩下样本中类别占多数的类别
3、从数据集中选取最佳的划分特征下标
4、根据最佳特征找到数据集和特征值相等的样本
5、根据最佳特征对数据集进行划分(根据特征值的不同可以划分出多个分支)
6、将划分结果记录到一个字典中
7、递归这个过程
'''
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): # 剩下所有样本都属于同一类别时，表示已到达叶子结点，返回该叶子结点的类别
        return classList[0]
    if len(dataSet[0]) == 1: # 当数据集遍历完所有特征时，仅剩下分类的类别，则选择类别最多的一个
        return majorityCnt(classList)

    bestFeatIndex = chooseBestFeatureToSplit(dataSet) # 从数据集中挑选最佳的划分特征
    besFeatureLabel = labels[bestFeatIndex] # 最佳特征的特征名
    del(labels[bestFeatIndex]) # 挑选后删除最佳特征，避免下次迭代时仍然选择同样的特征
    decisionTree = {besFeatureLabel: {}}
    bestFeatureValues = [example[bestFeatIndex] for example in dataSet]
    uniqueBestFeatureValues = set(bestFeatureValues)
    for value in uniqueBestFeatureValues:
        subLables = labels[:]
        splitData = splitDataSet(dataSet, bestFeatIndex, value)
        decisionTree[besFeatureLabel][value] = createTree(splitData, subLables)

    return decisionTree



def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

# def calcShannonEnt(dataSet):
#     numEntries = len(dataSet)
#     labelCounts = {}
#     for featVec in dataSet: #the the number of unique elements and their occurance
#         currentLabel = featVec[-1]
#         if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
#         labelCounts[currentLabel] += 1
#     shannonEnt = 0.0
#     for key in labelCounts:
#         prob = float(labelCounts[key])/numEntries
#         shannonEnt -= prob * log(prob,2) #log base 2
#     return shannonEnt


# 计算不同类别在该数据集中的信息熵，然后将各类别的信息熵相加，得出该数据集的信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        label = featVec[-1]
        if label not in labelCounts.keys():
            labelCounts[label] = 0
        labelCounts[label] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    returnDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:  # 当每条数据中某维度的数据等于预期值时，则将该条数据的该特征抽出来，剩下的数据加到划分的数据列表中
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:]) # [1, 2, 3].extend([4,5,6]) ==> [1,2,3,4,5,6]
            returnDataSet.append(reducedFeatVec) # [1, 2, 3].extend([4,5,6]) ==> [1,2,3,[4,5,6]]
    return returnDataSet

# 选择最好的特征进行数据划分，数据集中包含分类的类别
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet] # 遍历数据集，并将每个元素（每个元素是一个列表）的第一项取出，组成一个特征值列表
        # print featList
        uniqueVals = set(featList) # 将列表转化成集合，集合中的元素不存在重复的值
        # print uniqueVals
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet)) # 计算划分后的子集占总数据集的比例
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
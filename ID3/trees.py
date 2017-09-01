# coding=utf-8
# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
from math import log

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
        print featList
        uniqueVals = set(featList) # 将列表转化成集合，集合中的元素不存在重复的值
        print uniqueVals
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
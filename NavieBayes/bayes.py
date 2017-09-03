# coding=utf-8

from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec


# 创建一个包含在所有文档中出现的不重复词的列表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) # 求两个集合的并集
    return list(vocabSet)

# 词袋模型：每个单词可以出现多次
def bagOfWord2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print '%s is not in the vocabulary.' % word
    return returnVec


# 词集模型： 只将每个词出现与否作为一个特征
# 用列表记录输入的数据集中是否有词汇表中的词出现过
# 此函数可将输入的文档转化为算法输入所需的词向量
def setOfWord2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print '%s is not in the vocabulary.' % word  # 当测试样本中存在不属于词典的词时，不记录
    return returnVec

'''
trainDataSet: 训练数据集，是一个文档矩阵，每个元素是一个文档向量，长度和上面构造的词汇表一样，如果某个词出现了则记1，没出现则记为0
trainCategoryList: 由每篇文档类别标签所构成的向量

该方法可计算出各个词汇出现的概率，在对样本进行分类时，可用于计算样本的概率
'''
def trainNaviBayes0(trainDataSet, trainCategoryList):
    numTrainDocs = len(trainDataSet)
    numWords = len(trainDataSet[0])
    pAbusive = sum(trainCategoryList) / float(numTrainDocs)
    p0Num = ones(numWords) # 为避免出现一个单词的概率为0，其结果为0 的情况，将初始化的概率矩阵由0矩阵变为1矩阵，初始词汇总数由0设为2
    p0TotalWords = 2.0
    p1Num = ones(numWords)
    p1TotalWords = 2.0
    for i in range(numTrainDocs):
        #  求 p(wi | c0) 的值
        if trainCategoryList[i] == 0:  # 当该文档出现了侮辱性词语时（即分类为1的文档)
            p0Num += trainDataSet[i]  # 该文档各个词汇的频数都要加1
            p0TotalWords += sum(trainDataSet[i])    # 文档词汇总数也要相加
        else:
            # 求p(wi | c1)的值
            p1Num += trainDataSet[i]
            p1TotalWords += sum(trainDataSet[i])

    p0Vect = log(p0Num / p0TotalWords)  # 因为概率列表中的元素都很小，相乘后会很接近0，在python中会导致最后结果为0，因此对结果取对数，虽然大小改变了，但是两者的大小关系是不变的
    p1Vect = log(p1Num / p1TotalWords)

    return pAbusive, p0Vect, p1Vect


'''
这里的相乘是指对应元素相乘，而不是向量乘法。因为各个词元素的概率做了对数处理，因此:
原公式p1 = p(c1 | wi) = p(wi | c1) p(c1) = p(w0 | c1) p(w1 | c1) p(w2 | c1) ... p(wi | c1)p(c1)
两边取对数，得 log(p1) = log(p(w0 | c1) p(w1 | c1) p(w2 | c1) ... p(wi | c1)p(c1)) 
                    = log(p(w0 | c1)) + log(p(w1 | c1)) + log(p(w2 | c1)) + ... + log(p(wi | c1)) + log(p(c1))

在该函数中，testVec为测试的文档向量，如果存在记为1，不存在记为0，因此该向量和概率向量对应元素相乘，则刚好得到文档词汇的概率，然后各概率相加，再加上在某个类别下的概率的对数，即等于最终的概率的对数
概率取对数后不影响其大小关系的比较，因此可根据其大小关系得到样本的分类
'''
def NavieBayesClassify(testVec, p0Vec, p1Vec, pClass1):
    p1 = sum(testVec * p1Vec) + log(pClass1)
    p0 = sum(testVec * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

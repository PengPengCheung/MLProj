# coding=utf-8

import bayes
from numpy import *
import textHandler


def testNavieBayes(testVec):
    dataSet, classLabels = bayes.loadDataSet()
    vocabList = bayes.createVocabList(dataSet)
    trainMat = []
    for doc in dataSet:
        wordVec = bayes.setOfWord2Vec(vocabList, doc)
        trainMat.append(wordVec)

    pAb, p0V, p1V = bayes.trainNaviBayes0(array(trainMat), array(classLabels))
    testDoc = array(bayes.setOfWord2Vec(vocabList, testVec))  # 测试样本需要转化为词向量
    classLabel = bayes.NavieBayesClassify(testDoc, p0V, p1V, pAb)
    print classLabel


def spamTest():
    mailFileName = '/Users/peng/Files/TechFiles/机器学习实战及配套代码/machinelearninginaction/Ch04/email/'
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        textStr = open(mailFileName + 'spam/%d.txt' % i).read()
        wordList = textHandler.textParse(textStr)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        textStr = open(mailFileName + 'ham/%d.txt' % i).read()
        wordList = textHandler.textParse(textStr)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    # print docList
    vocabList = bayes.createVocabList(docList)
    # 划分测试集
    trainingSet = range(50)
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    # 构造训练集
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bayes.setOfWord2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    # 训练朴素贝叶斯分类器
    pSpam, p0V, p1V = bayes.trainNaviBayes0(array(trainMat), array(trainClasses))
    errorCount = 0

    # 对测试集进行测试
    for docIndex in testSet:
        wordVec = bayes.setOfWord2Vec(vocabList, docList[docIndex])
        if bayes.NavieBayesClassify(array(wordVec), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print docIndex, docList[docIndex]

    print 'the error rate is: ', float(errorCount) / len(testSet)




if __name__ == '__main__':
    # testEntry = ['stupid', 'garbage']
    # testNavieBayes(testEntry)
    spamTest()

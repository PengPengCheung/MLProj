# coding=utf-8

import storage
import trees

def handleDataSet(fileName):
    fr = open(fileName)
    lenses = [instance.strip().split('\t') for instance in fr.readlines()]
    # print lenses
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    storage.storeData(lensesLabels, 'DataLabels.txt')
    return lenses, lensesLabels


def trainTree(dataSet, labels):
    tree = trees.createTree(dataSet, labels)
    print tree
    storage.storeData(tree, 'classiferResult.txt')


def classifyDataSet(testVec):
    tree = storage.grabData('classiferResult.txt')
    labels = storage.grabData('DataLabels.txt')
    classLabel = trees.classify(tree, labels, testVec)
    return classLabel


def testTree(fileName, testVec):
    # dataSet, labels = handleDataSet(fileName)
    # trainTree(dataSet, labels)
    return classifyDataSet(testVec)

if __name__ == '__main__':
    fileName = '/Users/peng/Files/TechFiles/机器学习实战及配套代码/machinelearninginaction/Ch03/lenses.txt'
    testResult = testTree(fileName, ['pre', 'hyper', 'yes', 'reduced'])
    print testResult

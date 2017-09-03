# coding=utf-8

import trees
import storage


def trainTree(dataSet, labels):
    tree = trees.createTree(dataSet, labels)
    storage.storeData(tree, 'classiferResult.txt')


def handleDataSet():
    dataSet, labels = trees.createDataSet()
    storage.storeData(labels, 'DataLabels.txt')
    return dataSet, labels


def classifyDataSet(testVec):
    tree = storage.grabData('classiferResult.txt')
    labels = storage.grabData('DataLabels.txt')
    classLabel = trees.classify(tree, labels, testVec)
    return classLabel


if __name__ == '__main__':
    dataSet, labels = handleDataSet() # 数据预处理
    trainTree(dataSet, labels) # 训练数据并保存训练结果
    classLabel = classifyDataSet([1, 0]) # 测试分类结果
    print classLabel

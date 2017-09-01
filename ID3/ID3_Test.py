# coding=utf-8

import trees

if __name__ == '__main__':
    dataSet, labels = trees.createDataSet()
    shannonEnt = trees.calcShannonEnt(dataSet)
    # print dataSet, labels
    # print shannonEnt
    splitDataSet = trees.splitDataSet(dataSet, 0, 1)
    # print splitDataSet

    bestFeature = trees.chooseBestFeatureToSplit(dataSet)
    print bestFeature
# coding=utf-8

from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
import kNN2


def autoNumTest(dataSet):
    normal_data_set, ranges, minValues = kNN2.autoNum(dataSet)
    print normal_data_set
    return normal_data_set, ranges, minValues

def read_to_matrix_test():
    filename = '/Users/peng/Files/TechFiles/机器学习实战及配套代码/machinelearninginaction/Ch02/datingTestSet2.txt'
    datingDataMat, datingLabels = kNN2.file2matrix(filename)
    print datingDataMat[-1][0], datingDataMat[-1][1], datingDataMat[-1][2], datingLabels[-1]
    return datingDataMat, datingLabels

def matplotlib_test(x_array, y_array, datingLabels):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(x_array, y_array, s=15 * array(datingLabels),
               c=15 * array(datingLabels), marker='o')
    plt.xlabel('玩视频游戏所占时间百分比')
    plt.ylabel('每周消费的冰淇淋公升数')
    plt.show()

def kNNTest(normMat, labels, k):
    hotRatio = 0.1
    m = normMat.shape[0]
    numTestVecs = int(m * hotRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifyResult = kNN2.classify(normMat[i, :], normMat[numTestVecs:m, :], labels[numTestVecs:m], k) # k=3，选取3个最近邻的类
        if classifyResult != labels[i]:
            errorCount += 1
            print '第' + str(i) + '条数据的分类为：' + str(classifyResult) + ', 正确分类为：' + str(labels[i]) + ', 分类错误！ 已错误次数：' + str(errorCount)
        else:
            print '第' + str(i) + '条数据的分类为：' + str(classifyResult) + ', 正确分类为：' + str(labels[i]) + ', 分类正确！'

    accuracy_rate = 1-(errorCount / float(numTestVecs))
    print '分类正确率为：' + str(accuracy_rate)

    # for i in range(numTestVecs):
    #     classifierResult = kNN2.classify(normMat[i,] ^ normMat[numTestVecs];
    #     m,], \ datingLabels[numTestVecs:m], 3)
    #     # print "the classifier came back, with %d, the real answer is: %dH\ % (classifierResult, datingLabels[i])
    #     if (classifierResult != datingLabels[i]) errorCount += 1, 0
    #     print "the total error rate is: %f" % {errorCount / float(numTestVecs))

if __name__ == '__main__':
    datingDataMat, datingLabels = read_to_matrix_test()
    # matplotlib_test(datingDataMat[:, 1], datingDataMat[:, 2], datingLabels)
    normal_data_set, ranges, minValues = autoNumTest(datingDataMat)

    kNNTest(normal_data_set, datingLabels, 9)

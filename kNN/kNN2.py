# coding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from numpy import *
import operator


def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def autoNum(dataSet):
    minValues = dataSet.min(0)
    maxValues = dataSet.max(0)
    ranges = maxValues - minValues
    normalMat = zeros(shape(dataSet))
    row_len = dataSet.shape[0]
    # row_len = shape(dataSet)  ### shape(dataSet)  会计算出该矩阵的行数和列数。如果用dataSet.shape[0]则表示第0列的长度大小
    print row_len
    normalMat = dataSet - tile(minValues, (row_len, 1))
    normalMat = normalMat / tile(ranges, (row_len, 1))
    return normalMat, ranges, minValues


# kNN分类函数
def classify(input, dataSet, label, k):
    dataSize = dataSet.shape[0]  # 读取数据矩阵的维度，shape[0] 表示第一维度的长度

    ####计算欧式距离

    # tile函数
    # 第一个是矩阵A
    # 第二个参数是要
    # 只有一个数字时，表示
    # 对A中元素重复的次数
    # 两个参数时（x， y） y表示对A中元素重复的次数， x表示对前面的操作执行x次
    diff = tile(input, (dataSize, 1)) - dataSet
    sqdiff = diff ** 2
    squareDist = sum(sqdiff, axis=1)  ###行向量分别相加，从而得到新的一个行向量
    dist = squareDist ** 0.5

    ##对距离进行排序
    sortedDistIndex = argsort(dist)  ##argsort()根据元素的值从大到小对元素进行排序，返回下标

    classCount = {}
    for i in range(k):
        voteLabel = label[sortedDistIndex[i]]
        ###对选取的K个样本所属的类别个数进行统计
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    # sorted函数第一个参数指需要进行排序的数据集，第二个参数是比较器，key的值为一个函数，表示取数据集中的第二项进行比较，即比较classCount中的值，第三个值表示排序的顺序，reverse=True表示降序排序，从大到小排序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

    ###选取出现的类别次数最多的类别
    # maxCount = 0
    # for key, value in classCount.items():
    #     if value > maxCount:
    #         maxCount = value
    #         classes = key
    #
    # return classes


def file2matrix(filename):
    file_reader = open(filename)
    arrayOfLines = file_reader.readlines()
    print type(arrayOfLines)
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines, 3)) # 用0矩阵初始化行数为文件行数、列数为3的矩阵
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t') # 对每行数据以tab符进行切分，得到数据列表
        returnMat[index, :] = listFromLine[0:3] # 将每行的数据列表对应赋值到矩阵的每一行中
        classLabelVector.append(int(listFromLine[-1])) # -1的下标表示列表中的最后一个元素，即每条数据对应的类别
        index += 1

    return returnMat, classLabelVector



if __name__ == '__main__':
    dataSet, labels = create_data_set()
    input = array([1.1, 0.3])
    K = 3
    output = classify(input, dataSet, labels, K)
    print "测试数据为:", input, "分类结果为：", output
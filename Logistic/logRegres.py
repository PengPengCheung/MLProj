# coding=utf-8


from numpy import *
import matplotlib.pyplot as plt


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('/Users/peng/Files/TechFiles/机器学习实战及配套代码/machinelearninginaction/Ch05/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  # 此处在每个元素中加1，是为了后面的参数计算，常数项也可以有值相乘
        labelMat.append(int(lineArr[2]))

    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()  # 将类别标签转化为numpy矩阵并转置
    row, column = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((column, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error

    return weights


# 随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    row, column = shape(dataMatrix)
    alpha = 0.01
    weights = ones(column)  # 创建元素为1 的行向量
    for i in range(row):
        h = sigmoid(sum(dataMatrix[i] * weights))  # h是数值
        error = classLabels[i] - h  # error是数值
        weights = weights + alpha * error * dataMatrix[i]

    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=50):
    row, column = shape(dataMatrix)
    weights = ones(column)
    for j in range(numIter):
        dataIndex = range(row)
        for i in range(row):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del dataIndex[randIndex]

    return weights


# 传入的weights是一个行向量，array形式，列数和特征数相同
def plotBestFit(weights):
    # weights = wei.getA()  # 将矩阵本身以array的形式返回
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] # 获取矩阵行数，即获得点的数量
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1) # -3.0为起始点，3.0为终点，0.1为步长，返回一个从-3.0到3.0，步长为0.1的array，以此作为函数曲线x轴的起始点和终止点
    y = (-weights[0]-weights[1] * x) / weights[2] # 函数 w0 + w1 * x + w2 * y = 0, x和y分别代表x1和x2， 也恰好是坐标轴的表示。然后用y表示x，则刚好是这行代码的含义
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

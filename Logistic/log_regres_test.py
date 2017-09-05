# coding = utf-8

import logRegres
from numpy import *

if __name__ == '__main__':
    dataArr, labelMat = logRegres.loadDataSet()
    # weights = logRegres.gradAscent(dataArr, labelMat)
    weights = logRegres.stocGradAscent1(array(dataArr), labelMat)
    print weights
    logRegres.plotBestFit(weights)
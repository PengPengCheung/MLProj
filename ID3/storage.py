# coding=utf-8

import pickle


def storeData(inputData, fileName):
    fw = open(fileName, 'w')
    pickle.dump(inputData, fw)
    fw.close()


def grabData(fileName):
    fr = open(fileName)
    return pickle.load(fr)
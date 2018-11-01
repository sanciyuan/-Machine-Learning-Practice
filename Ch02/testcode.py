'''
datingTestSet2.txt
□ 每年获得的飞行常客里程数
□ 玩视频游戏所耗时间百分比
□ 每周消费的冰淇淋公升数
'''

#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from kNN import *
import matplotlib
import matplotlib.pyplot as plt
import time
# from kNN import createDataSet,classify0,file2matrix
from numpy import *
# from Digit_recog import *
from os import listdir

if __name__ == '__main__':


    # 测试数据
    group, labels = createDataSet()

    x = classify0([0, 0], group, labels, 3)
    print(x)

    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    print(datingDataMat, shape(datingDataMat), datingLabels)

    fig = plt.figure()
    # 在一张figure里面生成多张子图
    ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],15.0 * array(datingLabels), 15.0 * array(datingLabels))
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    plt.show()

    dating_mat, label_mat = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)

    datingClassTest()
    classfyPerson()

    testVec = img2vector('digits/testDigits/0_13.txt')
    print( testVec[0 , 0 : 31])
    print( testVec [ 0 , 32: 63])

    handwritingClassTest()
    # 这行代码耗时比较久，可以单独运行


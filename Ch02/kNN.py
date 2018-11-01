'''
Created on Oct 12, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most possible class label

@author: Peter Harrington
'''
#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import *
import operator
from os import listdir

def classify0(inX, dataSet, labels, k):
    '''kNN方法：输入测试数据inX、训练集dataSet、标签labels、k取值，返回预测标签'''
    # 数据集数量大小
    dataSetSize = dataSet.shape[0]
    # 距离计算
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  # tile()复制inX和dattaSet一样的维度
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    # 选择距离最小的k个点
    classCount={}          # k个点的标签计数器
    for i in range(k):
        # 取出前k个标签
        voteIlabel = labels[sortedDistIndicies[i]]
        # operator模块提供的itemgetter函数用于获取对象的哪些维的数据get(key, default=None) 函数返回指定键的值key，如果值不在字典中返回默认值default。
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    # reverse=True从大到小排序，operator模块提供的itemgetter函数用于获取对象的哪些维的数据
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createDataSet():
    '''测试knn的例子'''
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def file2matrix(filename):
    with open(filename) as fr:
        # 得到文件行数
        arrayOlines = fr.readlines()
        numberOfLines = len(arrayOlines)         #get the number of lines in the file
        # 创建返回的numpy矩阵
        returnMat = zeros((numberOfLines,3))        #prepare matrix to return
        classLabelVector = []                       #prepare labels return
        index = 0
        # 解析文件数据到列表中
        for line in arrayOlines:
            # 截取掉所有的回车字符
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index,:] = listFromLine[0:3]
            # 收集标签
            classLabelVector.append(int(listFromLine[-1]))
            index += 1
    return returnMat,classLabelVector
    
def autoNorm(dataSet):
    '''min(0)对数据集的列做归一化特征值，newValue={oldValue-min)/(max-min)'''
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    # print(errorCount)

def classfyPerson():
    '''约会网站预测函数'''
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("freguent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person：", resultList[classifierResult - 1])

def img2vector(filename):
    '''将图像转换为向量,把一个32*32的二进制图像矩阵转换为1*1024的向量'''
    returnVect = zeros((1,1024))
    with open(filename) as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    # 获取文件目录内的内容，以列表的形式返回文件名称['0_0.txt', '0_1.txt', '0_10.txt',...]
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    # 从文件名解析出分类数字
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))
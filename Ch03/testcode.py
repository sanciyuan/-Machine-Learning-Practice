#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from trees import *
from treePlotter import *


if __name__ == '__main__':
    myData, labels = createDataSet()
    print(myData)
    print(calcShannonEnt(myData))

    print(splitDataSet(myData, 0, 1))
    print(splitDataSet(myData, 0, 0))
    print(chooseBestFeatureToSplit(myData))

    myDat, labels = createDataSet()
    mytree = createTree(myData, labels)
    print(mytree)

    myTree = retrieveTree(0)
    print(getNumLeafs(myTree))
    print(getTreeDepth(myTree))

    myTree['no surfacing'][3] = 'maybe'
    createPlot(myTree)

    with open('lenses.txt') as fp:
        lenses = [line.strip().split('\t') for line in fp.readlines()]

    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']

    lense_Tree = createTree(lenses, lensesLabels)
    print(lense_Tree)
    createPlot(lense_Tree)
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    print(classify(lense_Tree, lensesLabels, ['young', 'hyper', 'yes', 'reduced']))


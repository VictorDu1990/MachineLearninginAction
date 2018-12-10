# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
from math import log
import operator
import pickle

def calcShannonEnt(dataSet):
	'''
	Function: calculate Shannon Entropy
	Parameter: dataSet - dataSets with labels
	Return: shannonEnt - Shannon Entropy of the dataSet
	Modify:
		2018-12-10
	'''
	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries
		shannonEnt -= prob * log(prob,2)
		
	return shannonEnt
	
def createDataSet():
	'''
	Function: create a simple datasets
	Parameter: none
	Return:
		dataSet - datasets
		labels - feature (what is the feature present?)labels 
	Modify:
		2018-12-10
	'''
	dataSet = [[0, 0, 0, 0, 'no'],						
			[0, 0, 0, 1, 'no'],
			[0, 1, 0, 1, 'yes'],
			[0, 1, 1, 0, 'yes'],
			[0, 0, 0, 0, 'no'],
			[1, 0, 0, 0, 'no'],
			[1, 0, 0, 1, 'no'],
			[1, 1, 1, 1, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[2, 0, 1, 2, 'yes'],
			[2, 0, 1, 1, 'yes'],
			[2, 1, 0, 1, 'yes'],
			[2, 1, 0, 2, 'yes'],
			[2, 0, 0, 0, 'no']]
	labels = ['Age', 'Havejob', 'Havehouse', 'Loan secured']
	return dataSet, labels 	


def splitDataSet(dataSet, axis, value):
	'''
	Function: split dataSet according to the given feature
	Parameter: 
		dataSet - dataSets
		axis - feature index
		value - feature value
	Return: 
		retDataSet - splited dataSet
	Modify:
		2018-12-10
	'''
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet
	

def chooseBestFeatureToSplit(dataSet):
	'''
	Function: choose the best feature
	Parameter: 
		dataSet - dataSets
	Return: 
		bestFeature - the index of  best feature(infoGain largest)
	Modify:
		2018-12-10
	'''
	numFeatures = len(dataSet[0]) - 1							#number of Features
	baseEntropy = calcShannonEnt(dataSet) 						#calculate Shannon Entropy
	bestInfoGain = 0.0  										#information gain
	bestFeature = -1											#the index of  best feature(infoGain largest)
	for i in range(numFeatures): 								#go through all features
		#the all i-th features
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)     					
		newEntropy = 0.0  								
		for value in uniqueVals: 						
			subDataSet = splitDataSet(dataSet, i, value) 		
			prob = len(subDataSet) / float(len(dataSet))   		
			newEntropy += prob * calcShannonEnt(subDataSet) 	
		infoGain = baseEntropy - newEntropy 						
		if (infoGain > bestInfoGain): 							#calculate the information gain
			bestInfoGain = infoGain 							#refresh the information gain
			bestFeature = i 									#record the index of feature
	return bestFeature 	
	
	
def majorityCnt(classList):
	'''
	Function: get the most class in the class list
	'''
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys():classCount[vote] = 0	
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)	
	return sortedClassCount[0][0]
	
def createTree(dataSet, labels, featLabels):
	'''
	Function: Create decision tree
	Parameter: 
		dataSet - dataSets
	Return: 
		bestFeature - the index of  best feature(infoGain largest)
	Modify:
		2018-12-10
	'''

	classList = [example[-1] for example in dataSet]			#get classification labels
	if classList.count(classList[0]) == len(classList):			#stop when all same class 
		return classList[0]
	if len(dataSet[0]) == 1 or len(labels) == 0:				#go through all features then return the most class
		return majorityCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataSet)				
	bestFeatLabel = labels[bestFeat]							
	featLabels.append(bestFeatLabel)
	myTree = {bestFeatLabel:{}}									#the tree
	del(labels[bestFeat])										#delete the label that has been used
	featValues = [example[bestFeat] for example in dataSet]		
	uniqueVals = set(featValues)								
	for value in uniqueVals:									#foreach features，Create decision tree					
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), labels, featLabels)
	return myTree
	
def getNumLeafs(myTree):
	'''
	get the total numbers of leaf node
	'''
    numLeafs = 0	
	# firstStr = list(myTree.keys())[0]
    firstStr = next(iter(myTree))								
    secondDict = myTree[firstStr]								
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':				
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs

def getTreeDepth(myTree):
	# get the tree depth 
    maxDepth = 0												
    firstStr = next(iter(myTree))								
    secondDict = myTree[firstStr]							
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':			
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth		
    return maxDepth
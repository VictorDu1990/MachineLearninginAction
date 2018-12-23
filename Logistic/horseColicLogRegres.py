# -*- coding:UTF-8 -*-
from sklearn.linear_model import LogisticRegression
import numpy as np
import random

def sigmoid(inX):
	return 1.0 / (1 + np.exp(-inX))

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
	'''
	Function: Improved Gradient ascent algorithm
	Parameters: 
		dataMatIn - dataset
		classLabels - labels
		numIter - Iteration times
	return:
		weights - Array of regression coefficients (optimal parameters)
		weights_array - Coefficient of regression for each update
	Modify:
		2018-12-23
	'''
	m,n = np.shape(dataMatrix)												
	weights = np.ones(n)   															
	for j in range(numIter):											
		dataIndex = list(range(m))
		for i in range(m):			
			alpha = 4/(1.0+j+i)+0.01   	 									
			randIndex = int(random.uniform(0,len(dataIndex)))				
			h = sigmoid(sum(dataMatrix[randIndex]*weights))					
			error = classLabels[randIndex] - h 								
			weights = weights + alpha * error * dataMatrix[randIndex]   	
			del(dataIndex[randIndex]) 										
	return weights 															



def gradAscent(dataMatIn, classLabels):
	'''
	Function: Gradient ascent algorithm
	Parameters: 
		dataMatIn - dataset
		classLabels - labels
	return:
		weights.getA() - Convert the matrix to an array and return the weight array
	Modify:
		2018-12-23
	'''
	dataMatrix = np.mat(dataMatIn)										
	labelMat = np.mat(classLabels).transpose()							
	m, n = np.shape(dataMatrix)											
	alpha = 0.01														
	maxCycles = 500														
	weights = np.ones((n,1))
	for k in range(maxCycles):
		h = sigmoid(dataMatrix * weights)								
		error = labelMat - h
		weights = weights + alpha * dataMatrix.transpose() * error
	return weights.getA()												



def colicTest():
	frTrain = open('horseColicTraining.txt')										#open trainingSet
	frTest = open('horseColicTest.txt')												#open testSet
	trainingSet = []; trainingLabels = []
	for line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(len(currLine)-1):
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[-1]))
	trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels,500)		#Improved Gradient ascent algorithm
	errorCount = 0; numTestVec = 0.0
	for line in frTest.readlines():
		numTestVec += 1.0
		currLine = line.strip().split('\t')
		lineArr =[]
		for i in range(len(currLine)-1):
			lineArr.append(float(currLine[i]))
		if int(classifyVector(np.array(lineArr), trainWeights))!= int(currLine[-1]):
			errorCount += 1
	errorRate = (float(errorCount)/numTestVec) * 100 								#errorRate
	print("Testset errorRate: %.2f%%" % errorRate)


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0


def colicSklearn():
	frTrain = open('horseColicTraining.txt')										#open trainingSet
	frTest = open('horseColicTest.txt')												#open testSet
	trainingSet = []; trainingLabels = []
	testSet = []; testLabels = []
	for line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(len(currLine)-1):
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[-1]))
	for line in frTest.readlines():
		currLine = line.strip().split('\t')
		lineArr =[]
		for i in range(len(currLine)-1):
			lineArr.append(float(currLine[i]))
		testSet.append(lineArr)
		testLabels.append(float(currLine[-1]))
	classifier = LogisticRegression(solver = 'sag',max_iter = 5000).fit(trainingSet, trainingLabels)
	test_accurcy = classifier.score(testSet, testLabels) * 100
	print('Accurcy:%f%%' % test_accurcy)

if __name__ == '__main__':
	colicSklearn()
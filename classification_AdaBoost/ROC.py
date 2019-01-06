# -*-coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def loadDataSet(fileName):
	numFeat = len((open(fileName).readline().split('\t')))
	dataMat = []; labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = []
		curLine = line.strip().split('\t')
		for i in range(numFeat - 1):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))

	return dataMat, labelMat

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
	"""
	Function: Single-level Decision Tree Classification Function
	Parameters:
		dataMatrix - data matrix
		dimen - the dimen-th column，the n-th features
		threshVal - threashold
		threshIneq - flag
	Returns:
		retArray - classify results
	Modify:
		2019-1-6
	"""
	retArray = np.ones((np.shape(dataMatrix)[0],1))				#Initialize the retArray as 1
	if threshIneq == 'lt':
		retArray[dataMatrix[:,dimen] <= threshVal] = -1.0	 	#If little than threshVal ,then set -1
	else:
		retArray[dataMatrix[:,dimen] > threshVal] = -1.0 		#If greater than threshVal ,then set -1
	return retArray
    

def buildStump(dataArr,classLabels,D):
	'''
	Function: Find the best Single-level Decision Tree Classification
	Parameters:
		dataArr - data matrix
		classLabels - data labels
		D - weights
	Returns:
		bestStump - the best Single-level Decision Tree
		minError - minimum error
		bestClasEst - best classify result
	'''
	dataMatrix = np.mat(dataArr); labelMat = np.mat(classLabels).T
	m,n = np.shape(dataMatrix)
	numSteps = 10.0; bestStump = {}; bestClasEst = np.mat(np.zeros((m,1)))
	minError = float('inf')														
	for i in range(n):															
		rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max()		
		stepSize = (rangeMax - rangeMin) / numSteps								#step
		for j in range(-1, int(numSteps) + 1): 									
			for inequal in ['lt', 'gt']:  										#lt:less than，gt:greater than
				threshVal = (rangeMin + float(j) * stepSize) 					#calculateing threashold value
				predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)#result 
				errArr = np.mat(np.ones((m,1))) 								#initialize the error Matrix
				errArr[predictedVals == labelMat] = 0 							#correct ,set 0
				weightedError = D.T * errArr  									#calculateing error
				# print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
				if weightedError < minError: 									#find the minimum error classifier
					minError = weightedError
					bestClasEst = predictedVals.copy()
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal
	return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
	''''
	Function: Using AdaBoost algorithm to improve the performance of weak classifier
	Parameters:
		dataArr - data matrix
		classLabels - label
		numIt - Maximum number of iterations
	Returns:
		weakClassArr - A well-trained classifier
		aggClassEst - Cumulative value of category estimation
	'''
	weakClassArr = []
	m = np.shape(dataArr)[0]
	D = np.mat(np.ones((m, 1)) / m)    										
	aggClassEst = np.mat(np.zeros((m,1)))
	for i in range(numIt):
		bestStump, error, classEst = buildStump(dataArr, classLabels, D) 	
		# print("D:",D.T)
		alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16))) 		
		bestStump['alpha'] = alpha  										
		weakClassArr.append(bestStump)                  					
		# print("classEst: ", classEst.T)
		expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst) 	
		D = np.multiply(D, np.exp(expon))                           		
		D = D / D.sum()														
		aggClassEst += alpha * classEst  									
		# print("aggClassEst: ", aggClassEst.T)
		aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m,1))) 	
		errorRate = aggErrors.sum() / m
		# print("total error: ", errorRate)
		if errorRate == 0.0: break 											# If Error 0, exit cycle
	return weakClassArr, aggClassEst

def plotROC(predStrengths, classLabels):

	font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
	cur = (1.0, 1.0) 														#Drawing cursor position
	ySum = 0.0 																#use for AUC caculateing
	numPosClas = np.sum(np.array(classLabels) == 1.0)						#Statistical number of positive classes
	yStep = 1 / float(numPosClas) 											#y-axis step	
	xStep = 1 / float(len(classLabels) - numPosClas) 						#x-axis step

	sortedIndicies = predStrengths.argsort() 								#Prediction intensity ranking, from low to high
	fig = plt.figure()
	fig.clf()
	ax = plt.subplot(111)
	for index in sortedIndicies.tolist()[0]:
		if classLabels[index] == 1.0:
			delX = 0; delY = yStep
		else:
			delX = xStep; delY = 0
			ySum += cur[1] 													#Heighth accumulated
		ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c = 'b') 	#plot ROC
		cur = (cur[0] - delX, cur[1] - delY) 								#Update the position of the drawing cursor
	ax.plot([0,1], [0,1], 'b--')
	plt.title('AdaBoostROC Curve of Equine Hernia Detection System', FontProperties = font)
	plt.xlabel('False positive rate', FontProperties = font)
	plt.ylabel('True positive rate', FontProperties = font)
	ax.axis([0, 1, 0, 1])
	print('AUC square:', ySum * xStep) 										#caculate AUC
	plt.show()
	

if __name__ == '__main__':
	dataArr, LabelArr = loadDataSet('horseColicTraining2.txt')
	weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, LabelArr, 50)
	plotROC(aggClassEst.T, LabelArr)
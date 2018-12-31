# -*- coding:UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np

def loadDataSet():
	# loadDataSet from file
	dataMat = []														
	labelMat = []														
	fr = open('testSet.txt')											
	for line in fr.readlines():											
		lineArr = line.strip().split()									
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])		
		labelMat.append(int(lineArr[2]))								
	fr.close()															
	return dataMat, labelMat											


def sigmoid(inX):
	# sigmoid() function
	return 1.0 / (1 + np.exp(-inX))


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
	alpha = 0.001														
	maxCycles = 500														
	weights = np.ones((n,1))
	for k in range(maxCycles):
		h = sigmoid(dataMatrix * weights)								
		error = labelMat - h
		weights = weights + alpha * dataMatrix.transpose() * error
	return weights.getA()					#Convert the matrix to an array and return the weight array


def plotDataSet():
	dataMat, labelMat = loadDataSet()									
	dataArr = np.array(dataMat)											
	n = np.shape(dataMat)[0]											
	xcord1 = []; ycord1 = []											
	xcord2 = []; ycord2 = []											
	for i in range(n):													
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])	
		else:
			xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])	
	fig = plt.figure()
	ax = fig.add_subplot(111)											
	ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)
	ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)			
	plt.title('DataSet')												
	plt.xlabel('X1'); plt.ylabel('X2')									
	plt.show()															


def plotBestFit(weights):
	dataMat, labelMat = loadDataSet()									
	dataArr = np.array(dataMat)											
	n = np.shape(dataMat)[0]											
	xcord1 = []; ycord1 = []											
	xcord2 = []; ycord2 = []											
	for i in range(n):													
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])	
		else:
			xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])	
	fig = plt.figure()
	ax = fig.add_subplot(111)											
	ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)
	ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)			
	x = np.arange(-3.0, 3.0, 0.1)
	y = (-weights[0] - weights[1] * x) / weights[2]
	ax.plot(x, y)
	plt.title('BestFit')												
	plt.xlabel('X1'); plt.ylabel('X2')									
	plt.show()		

if __name__ == '__main__':
	dataMat, labelMat = loadDataSet()	
	weights = gradAscent(dataMat, labelMat)
	plotBestFit(weights)
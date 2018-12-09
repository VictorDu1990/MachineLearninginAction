# -*- coding: UTF-8 -*-
import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN


def img2vector(filename):
	"""
	  Function: change [32x32] to [1x1024]
		filename - the file stored the datasets
	Returns:
		binnary image [1x1024] vector
	Modify:
		2018-12-8
	"""
	#create zeros vector[1x1024]
	returnVect = np.zeros((1, 1024))
	fr = open(filename)
	#read rows
	for i in range(32):
		#read one row
		lineStr = fr.readline()
		#32 element to returnVect
		for j in range(32):
			returnVect[0, 32*i+j] = int(lineStr[j])
	#[1x1024]
	return returnVect


def handwritingClassTest():
	"""
	Function: digits classifier tester
	Parameters:
	Returns:
	Modify:
		2018-12-9
	"""
	#Labels
	hwLabels = []
	# get file in trainingDigits
	trainingFileList = listdir('trainingDigits')
	#files number
	m = len(trainingFileList)
	#initialized testsets
	trainingMat = np.zeros((m, 1024))
	#get the label of training datasets
	for i in range(m):
		#filename
		fileNameStr = trainingFileList[i]
		#label
		classNumber = int(fileNameStr.split('_')[0])
		#add the label to hwLabels
		hwLabels.append(classNumber)
		#store the [1x1024] in trainingMat
		trainingMat[i,:] = img2vector('trainingDigits/%s' % (fileNameStr))

	#create kNN classifier
	neigh = kNN(n_neighbors = 3, algorithm = 'auto')
	#fit the kNN model
	neigh.fit(trainingMat, hwLabels)
	#files in dir testDigits
	testFileList = listdir('testDigits')
	#error counter
	errorCount = 0.0
	# how many test sets
	mTest = len(testFileList)
	#extract the testsets label and classification
	for i in range(mTest):
		#filename
		fileNameStr = testFileList[i]
		#get the number
		classNumber = int(fileNameStr.split('_')[0])
		#[1x1024]
		vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
		# predict results
		# classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		classifierResult = neigh.predict(vectorUnderTest)
		print("predict class: %d\tReal class: %d" % (classifierResult, classNumber))
		if(classifierResult != classNumber):
			errorCount += 1.0
	print("Total error num:%d\n Error ratio :%f%%" (errorCount, errorCount/mTest * 100))



if __name__ == '__main__':
	handwritingClassTest()

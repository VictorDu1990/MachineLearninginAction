# -*- coding: UTF-8 -*-
import numpy as np
import operator
from os import listdir
from sklearn.svm import SVC


def img2vector(filename):
	'''
	Function:change [32x32] binary image to [1x1024] vector。
	Parameters:
		filename - filename
	Returns:
		returnVect - binary image to [1x1024] vecto
	'''
	returnVect = np.zeros((1, 1024))
	#open file 
	fr = open(filename)
	#read lines
	for i in range(32):
		#each line
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0, 32*i+j] = int(lineStr[j])
	return returnVect

def handwritingClassTest():

	hwLabels = []
	trainingFileList = listdir('trainingDigits')
	m = len(trainingFileList)
	trainingMat = np.zeros((m, 1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		classNumber = int(fileNameStr.split('_')[0])
		hwLabels.append(classNumber)
		trainingMat[i,:] = img2vector('trainingDigits/%s' % (fileNameStr))
	clf = SVC(C=200,kernel='rbf')
	clf.fit(trainingMat,hwLabels)
	testFileList = listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		classNumber = int(fileNameStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
		#get predict result
		# classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		classifierResult = clf.predict(vectorUnderTest)
		print("Classification result：%d\t real result:%d" % (classifierResult, classNumber))
		if(classifierResult != classNumber):
			errorCount += 1.0
	print("Total error: %d \nError rate:%f%%" % (errorCount, errorCount/mTest * 100))

if __name__ == '__main__':
	handwritingClassTest()
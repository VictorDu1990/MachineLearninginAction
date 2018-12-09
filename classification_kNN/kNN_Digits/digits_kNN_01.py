# -*- coding: UTF-8 -*-
import numpy as np
import operator
from os import listdir

def classify0(inX, dataSet, labels, k):
  """
	Function: kNN algorithm classifier
	Parameters:
		inX - testing datasets
		dataSet - training datasets
		labes - classes of the datasets
		k - kNN parameter, choose the minimum k points 
	Returns:
		sortedClassCount[0][0] - classification result
	Modify:
		2018-12-8
	"""
  
  #get the total row number of dataSet
	dataSetSize = dataSet.shape[0]
	#repeating inX the number of times rows=dataSetSize, columns=1
	diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
	#squared the difference
	sqDiffMat = diffMat**2
	#sum() for all elements,sum(0) for columns,sum(1) for rows
	sqDistances = sqDiffMat.sum(axis=1)
	#calculate distance
	distances = sqDistances**0.5
	#return the inexes after descend sorted the distances elements
	sortedDistIndices = distances.argsort()
	#define a dict to record the counts of classifications
	classCount = {}
	for i in range(k):
		#get the first k classifications
		voteIlabel = labels[sortedDistIndices[i]]
		#calculate the count of each classifications
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

	sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	print(sortedClassCount)
	#Return the  category that the most times, that is, the final category.
	return sortedClassCount[0][0]



def file2matrix(filename):
  """
  Function:Open and parse files and classify the data. 1 for dislike, 2 for smallDoses, 3 for largeDoses
  Parameters:
    filename - the file stored the datasets
  Returns:
    returnMat - the features  matrix
    classLabelVector - class_label vector
  Modify:
    2018-12-8
  """
	# open datasets file
    	fr = open(filename,'r',encoding = 'utf-8')
	#Read all the contents of the file
	arrayOLines = fr.readlines()
	#remove BOM if it does have.
  	arrayOLines[0]=arrayOLines[0].lstrip('\ufeff')
	#get the lines numbers of the file
	numberOfLines = len(arrayOLines)

	returnMat = np.zeros((numberOfLines,3))
	classLabelVector = []
	# row index 
	index = 0
	for line in arrayOLines:
		#s.strip() :Default deletion of blank characters(include '\n','\r','\t',' ')
		line = line.strip()
		listFromLine = line.split('\t')
		#get feature matrix
		returnMat[index,:] = listFromLine[0:3]
		# label it: 1 for dislike, 2 for smallDoses, 3 for largeDoses
		if listFromLine[-1] == 'didntLike':
			classLabelVector.append(1)
		elif listFromLine[-1] == 'smallDoses':
			classLabelVector.append(2)
		elif listFromLine[-1] == 'largeDoses':
			classLabelVector.append(3)
		index += 1
	return returnMat, classLabelVector


def img2vector(filename):
  """
  Function:Open and parse files and classify the data. 1 for dislike, 2 for smallDoses, 3 for largeDoses
  Parameters:
    filename - the file stored the datasets
  Returns:
    returnVect - vector 1x1024 
  Modify:
    2018-12-8
  """
	#create a [1x1024] zeros vector
	returnVect = np.zeros((1, 1024))
	#open data file
	fr = open(filename)
	#read per row
	for i in range(32):
		#read one row
		lineStr = fr.readline()
		
		for j in range(32):
			returnVect[0, 32*i+j] = int(lineStr[j])
	#reurrn the [1x1024] vector
	return returnVect

"""
Function:digits classifier tester
Parameters:none
Returns:none
Modify:
	2018-12-9
"""
def handwritingClassTest():
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
	#in testDigits filenames
	testFileList = listdir('testDigits')
	#error counter
	errorCount = 0.0
	#number of test datasets
	mTest = len(testFileList)
	#get  the label of testsets and test it
	for i in range(mTest):
		fileNameStr = testFileList[i]
		classNumber = int(fileNameStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
		#predict results
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		print("predict class: %d\tReal class: %d" % (classifierResult, classNumber))
		if(classifierResult != classNumber):
			errorCount += 1.0
	print("Total error num:%d\n Error ratio :%f%%" % (errorCount, errorCount/mTest))


if __name__ == '__main__':
	handwritingClassTest()


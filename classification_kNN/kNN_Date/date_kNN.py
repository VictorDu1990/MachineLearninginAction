# -*- coding: UTF-8 -*-

from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import operator



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

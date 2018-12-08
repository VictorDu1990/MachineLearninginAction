# -*- coding: UTF-8 -*-
import numpy as np
import operator
import collections

def createDataSet():
  """
  Function: to create datasets
  Parameters:	None
  Returns: group - the datasets
    			 labels - the real labels of the datasets
  Modify: 2018-12-8
  """
  #features
	group = np.array([[1.0, 1.1],[1.0, 1.0],[0.0, 0.0],[0.0, 0.1]])
	#the label corresponding to the features 
	labels = ['A','A','B','B']
	return group, labels


def classify0(inx, dataset, labels, k):
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
	# calculate distance
	dist = np.sum((inx - dataset)**2, axis=1)**0.5
	# the k nearest labels
	k_labels = [labels[index] for index in dist.argsort()[0 : k]]
	
	# The label that appears most frequently is the final category
	label = collections.Counter(k_labels).most_common(1)[0][0]
	return label

if __name__ == '__main__':
	#create datasets
	group, labels = createDataSet()
	#test dataset
	test = [1.1, 1.2]
	#kNN classification
	test_class = classify0(test, group, labels, 3)
	#print the classification result
	print(test_class)

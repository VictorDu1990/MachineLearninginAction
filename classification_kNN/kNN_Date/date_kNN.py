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

def showdatas(datingDataMat, datingLabels):
	
	"""
	Function:Data visualization
	Parameters:
		datingDataMat - the features matrix
		datingLabels - classifications Labels
	Returns:
		none
	Modify:
		2018-12-9
	"""
	fig, axs = plt.subplots(nrows=2, ncols=2,sharex=False, sharey=False, figsize=(13,8))

	numberOfLabels = len(datingLabels)
	LabelsColors = []
	for i in datingLabels:
		if i == 1:
			LabelsColors.append('black')
		if i == 2:
			LabelsColors.append('orange')
		if i == 3:
			LabelsColors.append('red')
	#plot scatter
	axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelsColors,s=15, alpha=.5)
	#set title,x_axis label,y_axis label
	axs0_title_text = axs[0][0].set_title('Flight mileage obtained annually && Time consumption ratio of video games(%)')
	axs0_xlabel_text = axs[0][0].set_xlabel('Flight mileage obtained annually(km)')
	axs0_ylabel_text = axs[0][0].set_ylabel('Time consumption ratio of video games(%)')
	plt.setp(axs0_title_text, size=9, weight='bold', color='red')  
	plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')  
	plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black') 

	#plot scatter
	axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
	#set title,x_axis label,y_axis label
	axs1_title_text = axs[0][1].set_title('Flight mileage obtained annually(km) && Ice Cream Consumption Weekly(L)')
	axs1_xlabel_text = axs[0][1].set_xlabel('Flight mileage obtained annually(km)')
	axs1_ylabel_text = axs[0][1].set_ylabel('Ice Cream Consumption Weekly(L)')
	plt.setp(axs1_title_text, size=9, weight='bold', color='red')  
	plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')  
	plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black') 

	#plot scatter
	axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
	#set title,x_axis label,y_axis label
	axs2_title_text = axs[1][0].set_title('Time consumption ratio of video games(%) && Ice Cream Consumption Weekly(L)')
	axs2_xlabel_text = axs[1][0].set_xlabel('Time consumption ratio of video games(%)',)
	axs2_ylabel_text = axs[1][0].set_ylabel('Ice Cream Consumption Weekly(L)')
	plt.setp(axs2_title_text, size=9, weight='bold', color='red')  
	plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')  
	plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black') 
	#set legend
	didntLike = mlines.Line2D([], [], color='black', marker='.',
                      markersize=6, label='didntLike')
	smallDoses = mlines.Line2D([], [], color='orange', marker='.',
	                  markersize=6, label='smallDoses')
	largeDoses = mlines.Line2D([], [], color='red', marker='.',
	                  markersize=6, label='largeDoses')
	#add legend
	axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
	axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
	axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
	#show the image
	plt.show()


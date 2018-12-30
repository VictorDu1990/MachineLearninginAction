# -*-coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random


class optStruct:
	'''
	Function: Data structure, maintaining all the values that need to be manipulated
	Parameters：
		dataMatIn - data matrix
		classLabels - labels
		C - Slack variable
		toler - Fault tolerance rate
	Modify:
		2018-12-30
	'''
	def __init__(self, dataMatIn, classLabels, C, toler):
		self.X = dataMatIn								#data matrix
		self.labelMat = classLabels						#labels
		self.C = C 										#Slack variable
		self.tol = toler 								#Fault tolerance rate
		self.m = np.shape(dataMatIn)[0] 				#row number
		self.alphas = np.mat(np.zeros((self.m,1))) 		#initialize alpha as 0	
		self.b = 0 										#initialize b as 0
		self.eCache = np.mat(np.zeros((self.m,2))) 		。

def loadDataSet(fileName):

	dataMat = []; labelMat = []
	fr = open(fileName)
	for line in fr.readlines():                                     
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]), float(lineArr[1])])
		labelMat.append(float(lineArr[2]))                          
	return dataMat,labelMat

def calcEk(oS, k):
	"""
	Function:
		calculate error
	Parameters：
		oS - Data structure
		k - data label k
	Returns:
	    Ek - Data error labeled K
	"""
	fXk = float(np.multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T) + oS.b)
	Ek = fXk - float(oS.labelMat[k])
	return Ek

def selectJrand(i, m):

	#Function Description: Random Selection of Index Value of alpha_j
	j = i                                
	while (j == i):
		j = int(random.uniform(0, m))
	return j

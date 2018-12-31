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
		kTup - A tuple containing the information of a kernel function. 
			The first parameter stores the class of the kernel function, 
			and the second parameter stores the parameters needed by the necessary kernel function.
	Modify:
		2018-12-30
	'''
	def __init__(self, dataMatIn, classLabels, C, toler, kTup):
		self.X = dataMatIn								#data matrix
		self.labelMat = classLabels						#labels
		self.C = C 										#Slack variable
		self.tol = toler 								#Fault tolerance rate
		self.m = np.shape(dataMatIn)[0] 				#row number
		self.alphas = np.mat(np.zeros((self.m,1))) 		#initialize alpha as 0	
		self.b = 0 										#initialize b as 0
		self.eCache = np.mat(np.zeros((self.m,2))) 		#The tiger error buffer is initialized according to the number of rows in the matrix. 
														#The first is the valid flag bit, and the second is the actual value of error E.
		self.K = np.mat(np.zeros((self.m,self.m)))		
		for i in range(self.m):							
			self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

def kernelTrans(X, A, kTup): 
	'''
	Function: Transforming data into higher dimensional space through kernel function
	Parameters：
		X - data matrix
		A - a data vector
		kTup - Tuples containing kernel function information
	Returns:
	    K - the calculated core K
	'''
	m,n = np.shape(X)
	K = np.mat(np.zeros((m,1)))
	if kTup[0] == 'lin': K = X * A.T   					#Linear Kernel Function, only the inner product.
	elif kTup[0] == 'rbf': 								#Gauss Kernel Function, Calculated according to Gauss Kernel Function Formula
		for j in range(m):
			deltaRow = X[j,:] - A
			K[j] = deltaRow*deltaRow.T
		K = np.exp(K/(-1*kTup[1]**2)) 					#Computation of Gauss Kernel K
	else: raise NameError('Kernel function unrecognizable')
	return K 											#Returns the calculated core K

def loadDataSet(fileName):

	dataMat = []; labelMat = []
	fr = open(fileName)
	for line in fr.readlines():                                     
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]), float(lineArr[1])])
		labelMat.append(float(lineArr[2]))                          
	return dataMat,labelMat

def calcEk(oS, k):
	'''
	Function:
		calculate error
	Parameters：
		oS - Data structure
		k - data label k
	Returns:
	    Ek - Data error labeled K
	Modify:
		2018-12-31
	'''
	fXk = float(np.multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T) + oS.b)
	Ek = fXk - float(oS.labelMat[k])
	return Ek

def selectJrand(i, m):
	#Function Description: Random Selection of Index Value of alpha_j
	j = i                                
	while (j == i):
		j = int(random.uniform(0, m))
	return j


def selectJ(i, oS, Ei):
	'''
	Function: Inner-loop heuristics method
	Parameters：
		i - index
		oS - data structure
		Ei - error of i
	Returns:
	    j, maxK - index
	    Ej - Data error labeled J
	'''
	maxK = -1; maxDeltaE = 0; Ej = 0 						
	oS.eCache[i] = [1,Ei]  									
	validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]		
	if (len(validEcacheList)) > 1:							
		for k in validEcacheList:   						
			if k == i: continue 							
			Ek = calcEk(oS, k)								
			deltaE = abs(Ei - Ek)							
			if (deltaE > maxDeltaE):						
				maxK = k; maxDeltaE = deltaE; Ej = Ek
		return maxK, Ej										
	else:   												
		j = selectJrand(i, oS.m)							
		Ej = calcEk(oS, j)									
	return j, Ej 											

def updateEk(oS, k):
	#Calculate Ek and update error cache
	Ek = calcEk(oS, k)										
	oS.eCache[k] = [1,Ek]									

def clipAlpha(aj,H,L):
	if aj > H: 
		aj = H
	if L > aj:
		aj = L
	return aj
def innerL(i, oS):
	'''
	Function: Optimized SMO algorithm
	Parameters：
		i - data index that labeled i
		oS - data structure
	Returns:
		1 - Any pair of alpha values change
		0 - No one pair of alpha values changed or changed too little.
	'''
	#Step 1: Calculate error Ei
	Ei = calcEk(oS, i)
	#Optimize alpha and set a certain fault tolerance rate.
	if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
		#Select alpha_j using inner loop heuristic 2 and calculate Ej
		j,Ej = selectJ(i, oS, Ei)
		#Save the aplpha value before updating, using deep copy
		alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
		#Step 2: Compute upper and lower bounds L and H
		if (oS.labelMat[i] != oS.labelMat[j]):
			L = max(0, oS.alphas[j] - oS.alphas[i])
			H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
		else:
			L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
			H = min(oS.C, oS.alphas[j] + oS.alphas[i])
		if L == H: 
			print("L==H")
			return 0
		#Step 3: Calculate ETA
		eta = 2.0 * oS.X[i,:] * oS.X[j,:].T - oS.X[i,:] * oS.X[i,:].T - oS.X[j,:] * oS.X[j,:].T
		if eta >= 0: 
			print("eta>=0")
			return 0
		#Step 4: Update alpha_j
		oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej)/eta
		#Step 5: Trim alpha_j
		oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
		#Update Ej to Error Cache
		updateEk(oS, j)
		if (abs(oS.alphas[j] - alphaJold) < 0.00001): 
			print("alpha_j change too small.")
			return 0
		#Step 6: Update alpha_i
		oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
		#Update Ei to Error Cache
		updateEk(oS, i)
		#Step 7: Update b_1 and b_2
		b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
		b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
		#Step 8: Update B according to b_1 and b_2
		if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
		elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
		else: oS.b = (b1 + b2)/2.0
		return 1
	else: 
		return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter):
	'''
	Function: Complete Linear SMO Algorithms
	Parameters：
		dataMatIn - matrix
		classLabels - label
		C - slack variable
		toler - tolerance error 
		maxIter - Max Iteration times
	Returns:
		oS.b - SMO get the b
		oS.alphas - SMO get the alphas
	Modify:
		2018-12-31
	'''
	oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)		#Initialization of data structures
	iter = 0 																			#Initialize the current iteration number
	entireSet = True; alphaPairsChanged = 0
	while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):				#If alpha does not update or exceed the maximum number of iterations throughout the data set, exit the loop
		alphaPairsChanged = 0
		if entireSet:																	#Traversing the entire data set   						
			for i in range(oS.m):        
				alphaPairsChanged += innerL(i,oS)										#Using optimized SMO algorithm
				print("Full sample traversal:the %d Iteration sample:%d, alpha optimize times:%d" % (iter,i,alphaPairsChanged))
			iter += 1
		else: 																			#Non-boundary traversal
			nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]			#Traversing alpha that is not at boundaries 0 and C
			for i in nonBoundIs:
				alphaPairsChanged += innerL(i,oS)
				print("Non-boundary traversal:the %d Iteration sample:%d, alpha optimize times:%d" % (iter,i,alphaPairsChanged))
			iter += 1
		if entireSet:																	#Change to non-boundary traversal after one traversal
			entireSet = False
		elif (alphaPairsChanged == 0):													#If alpha is not updated, calculate full sample traversal 
			entireSet = True  
		print("Iteration times: %d" % iter)
	return oS.b,oS.alphas 																#Return b and alphas calculated by SMO algorithm


def img2vector(filename):
	#Converting 32x32 binary image to 1x1024 vector。

	returnVect = np.zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0,32*i+j] = int(lineStr[j])
	return returnVect

def loadImages(dirName):
	# load image	
	from os import listdir
	hwLabels = []
	trainingFileList = listdir(dirName)           
	m = len(trainingFileList)
	trainingMat = np.zeros((m,1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]     
		classNumStr = int(fileStr.split('_')[0])
		if classNumStr == 9: hwLabels.append(-1)
		else: hwLabels.append(1)
		trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
	return trainingMat, hwLabels    

def testDigits(kTup=('rbf', 10)):

	dataArr,labelArr = loadImages('trainingDigits')
	b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10, kTup)
	datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
	svInd = np.nonzero(alphas.A>0)[0]
	sVs=datMat[svInd] 
	labelSV = labelMat[svInd];
	print("Number of Support Vectors:%d" % np.shape(sVs)[0])
	m,n = np.shape(datMat)
	errorCount = 0
	for i in range(m):
		kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
		predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
		if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1
	print("Test Set Error Rate: %.2f%%" % (float(errorCount)/m))
	dataArr,labelArr = loadImages('testDigits')
	errorCount = 0
	datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
	m,n = np.shape(datMat)
	for i in range(m):
		kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
		predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
		if np.sign(predict) != np.sign(labelArr[i]): errorCount += 1    
	print("Test Set Error Rate: %.2f%%" % (float(errorCount)/m))

if __name__ == '__main__':
	testDigits()
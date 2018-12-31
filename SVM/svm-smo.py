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
		self.eCache = np.mat(np.zeros((self.m,2))) 		

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


def showClassifer(dataMat, classLabels, w, b):
	'''
	Function:Visualization of classification results
	Parameters:
		dataMat - matrix
	    w - Linear Normal Vector
	    b - intercept
	Returns:
	    none
	Modify:
		2018-12-31
	'''
	#plot the samples
	data_plus = []                                  
	data_minus = []                                 
	for i in range(len(dataMat)):
		if classLabels[i] > 0:
			data_plus.append(dataMat[i])
		else:
			data_minus.append(dataMat[i])
	data_plus_np = np.array(data_plus)              
	data_minus_np = np.array(data_minus)            
	plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)   
	plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7) 
	#plot the straight line
	x1 = max(dataMat)[0]
	x2 = min(dataMat)[0]
	a1, a2 = w
	b = float(b)
	a1 = float(a1[0])
	a2 = float(a2[0])
	y1, y2 = (-b- a1*x1)/a2, (-b - a1*x2)/a2
	plt.plot([x1, x2], [y1, y2])
	#Finding Support Vector Points
	for i, alpha in enumerate(alphas):
		if alpha > 0:
			x, y = dataMat[i]
			plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
	plt.show()


def calcWs(alphas,dataArr,classLabels):
	'''
	Funtion: get the w
	Parameters:
		dataArr - data matrix
	    classLabels - labels
	    alphas - alphas values
	Returns:
	    w - the w that calculated
	'''
	X = np.mat(dataArr); labelMat = np.mat(classLabels).transpose()
	m,n = np.shape(X)
	w = np.zeros((n,1))
	for i in range(m):
		w += np.multiply(alphas[i]*labelMat[i],X[i,:].T)
	return w

if __name__ == '__main__':
	dataArr, classLabels = loadDataSet('testSet.txt')
	b, alphas = smoP(dataArr, classLabels, 0.6, 0.001, 40)
	w = calcWs(alphas,dataArr, classLabels)
	showClassifer(dataArr, classLabels, w, b)
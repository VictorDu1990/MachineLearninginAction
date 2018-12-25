# -*- coding:UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random


def loadDataSet(fileName):
	# load DataSet from file
	dataMat = []; labelMat = []
	fr = open(fileName)
	for line in fr.readlines():                                     
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]), float(lineArr[1])])      
		labelMat.append(float(lineArr[2]))                          
	return dataMat,labelMat



def selectJrand(i, m):
	# select a j randomly that j!=i
	j = i                                 
	while (j == i):
		j = int(random.uniform(0, m))
	return j


def clipAlpha(aj,H,L):
	# modify aj 
	if aj > H: 
		aj = H
	if L > aj:
		aj = L
	return aj


def showDataSet(dataMat, labelMat):
	# virtualize the dataset
	data_plus = []                            # + sample	
	data_minus = []                           # - sample
	for i in range(len(dataMat)):
		if labelMat[i] > 0:
			data_plus.append(dataMat[i])
		else:
			data_minus.append(dataMat[i])
	data_plus_np = np.array(data_plus)              
	data_minus_np = np.array(data_minus)            
	plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])   #+ sample scatter
	plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1]) #- sample scatter
	plt.show()

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
	'''
	Function: Simplified SMO Algorithms
	Parameters:
		DataMatIn - Data Matrix
		ClassLabels - Data Labels
		C - relaxation variable
		Toler - fault tolerance rate
		MaxIter - Maximum number of iterations
	Returns: none
	Modify:
		2018-12-25
	'''
	dataMatrix = np.mat(dataMatIn); labelMat = np.mat(classLabels).transpose()

	b = 0; 
	m,n = np.shape(dataMatrix)
	alphas = np.mat(np.zeros((m,1)))
	iter_num = 0
	while (iter_num < maxIter):
		alphaPairsChanged = 0
		for i in range(m):
			# step 1 ：calculation error Ei
			fXi = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
			Ei = fXi - float(labelMat[i])
			#Optimize alpha and set a certain fault tolerance rate.
			if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
				#Random selection of another alpha_j optimized in pairs with alpha_i
				j = selectJrand(i,m)
				#step 1：calculation error Ej
				fXj = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
				Ej = fXj - float(labelMat[j])
				#Save the aplpha value before updating, using deep copy
				alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
				#step 2：Computing upper and lower bounds L and H
				if (labelMat[i] != labelMat[j]):
				    L = max(0, alphas[j] - alphas[i])
				    H = min(C, C + alphas[j] - alphas[i])
				else:
				    L = max(0, alphas[j] + alphas[i] - C)
				    H = min(C, alphas[j] + alphas[i])
				if L==H: print("L==H"); continue
				#step 3：calculation eta
				eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
				if eta >= 0: print("eta>=0"); continue
				#step 4：update alpha_j
				alphas[j] -= labelMat[j]*(Ei - Ej)/eta
				#step 5：modify alpha_j
				alphas[j] = clipAlpha(alphas[j],H,L)
				if (abs(alphas[j] - alphaJold) < 0.00001): print("alpha_j变化太小"); continue
				#step 6：update alpha_i
				alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
				#step 7：update b_1 and b_2
				b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
				b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
				#step 8：according to b_1 and b_2 to update b
				if (0 < alphas[i]) and (C > alphas[i]): b = b1
				elif (0 < alphas[j]) and (C > alphas[j]): b = b2
				else: b = (b1 + b2)/2.0
				#Statistical optimization times
				alphaPairsChanged += 1
				#Print statistics
				print("the %d-th iterations Sample:%d, alpha iterations num:%d" % (iter_num,i,alphaPairsChanged))
		#update iterations num
		if (alphaPairsChanged == 0): iter_num += 1
		else: iter_num = 0
		print("iterations num: %d" % iter_num)
	return b,alphas

def showClassifer(dataMat, w, b):
	#Function description: visualization of classification results	  
	data_plus = []                                   
	data_minus = []                                  
	for i in range(len(dataMat)):
		if labelMat[i] > 0:
			data_plus.append(dataMat[i])
		else:
			data_minus.append(dataMat[i])
	data_plus_np = np.array(data_plus)               
	data_minus_np = np.array(data_minus)            
	plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)   #+ sample scatter
	plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7) #- sample scatter
	#plot line
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
		if abs(alpha) > 0:
			x, y = dataMat[i]
			plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
	plt.show()



def get_w(dataMat, labelMat, alphas):
    alphas, dataMat, labelMat = np.array(alphas), np.array(dataMat), np.array(labelMat)
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
    return w.tolist()


if __name__ == '__main__':
	dataMat, labelMat = loadDataSet('data/testSet.txt')
	b,alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
	w = get_w(dataMat, labelMat, alphas)
	showClassifer(dataMat, w, b)
	
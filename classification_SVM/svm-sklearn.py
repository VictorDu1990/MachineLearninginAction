# -*-coding:utf-8 -*-
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

def loadDataSet(filename):
	with open("test.txt","r") as file:
		ty=-3 				#get the label column index
		result=[]
		for line in file.readlines():
			result.append(list(map(str,line.strip().split(','))))

		vec = np.array(result)
		dataMat = vec[:,:-3]		#get the features
		labelMat = vec[:,ty]		#label 		
	return dataMat,labelMat
	
if __name__ == '__main__':
	dataMat,labelMat = loadDataSet("test.txt")
	# Dividing Test Sets and Training Sets	
	train_x,test_x,train_y,test_y = train_test_split(dataMat,labelMat,test_size=0.2)
    #Model training and prediction
    clf = SVC(kernel='linear',C=0.4)
    clf.fit(train_x,train_y)
    
    pred_y = clf.predict(test_x)
    print(classification_report(test_y,pred_y))
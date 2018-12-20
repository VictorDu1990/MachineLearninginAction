# -*- coding: UTF-8 -*-
import numpy as np
import random
import re



def createVocabList(dataSet):
    vocabSet = set([])  					
    for document in dataSet:				
        vocabSet = vocabSet | set(document) 
    return list(vocabSet)

"""
函数说明:根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0

Parameters:
	vocabList - createVocabList返回的列表
	inputSet - 切分的词条列表
Returns:
	returnVec - 文档向量,词集模型
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-11
"""
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)									
    for word in inputSet:												
        if word in vocabList:											
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec													



def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)										
    for word in inputSet:												
        if word in vocabList:											
            returnVec[vocabList.index(word)] += 1
    return returnVec													


def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)							
    numWords = len(trainMatrix[0])							
    pAbusive = sum(trainCategory)/float(numTrainDocs)		
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)	
    p0Denom = 2.0; p1Denom = 2.0                        	
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:							
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:												
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)							
    p0Vect = np.log(p0Num/p0Denom)          
    return p0Vect,p1Vect,pAbusive							


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)    	
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

def textParse(bigString):                                                   
    listOfTokens = re.split(r'\W*', bigString)                              
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]            


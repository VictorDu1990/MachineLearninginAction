# -*- coding: UTF-8 -*-
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import os
import random
import jieba

def TextProcessing(folder_path, test_size = 0.2):
	'''
	Function: Chinese text processing

	Parameters:
		folder_path - file path
		test_size - Test set percentage, default value =20% 
	Returns:
		all_words_list - List of training sets sorted in descending order of word frequency
		train_data_list - train dataset list
		test_data_list - testset label
		train_class_list - train dataset label list 
		test_class_list - testset label list
	Modify:
		2018-12-22
	'''
	folder_list = os.listdir(folder_path)						
	data_list = []												#datasets
	class_list = []												#labels

	# Traverse through each subfolder
	for folder in folder_list:
		new_folder_path = os.path.join(folder_path, folder)		
		files = os.listdir(new_folder_path)						
		
		j = 1
		#Traverse through each TXT file
		for file in files:
			if j > 100:											
				break
			with open(os.path.join(new_folder_path, file), 'r', encoding = 'utf-8') as f:	
				raw = f.read()
			
			word_cut = jieba.cut(raw, cut_all = False)			
			word_list = list(word_cut)							
			
			data_list.append(word_list)							
			class_list.append(folder)							
			j += 1

	data_class_list = list(zip(data_list, class_list))			
	random.shuffle(data_class_list)								
	index = int(len(data_class_list) * test_size) + 1			
	train_list = data_class_list[index:]						
	test_list = data_class_list[:index]							
	train_data_list, train_class_list = zip(*train_list)		
	test_data_list, test_class_list = zip(*test_list)			

	all_words_dict = {}											
	for word_list in train_data_list:
		for word in word_list:
			if word in all_words_dict.keys():
				all_words_dict[word] += 1
			else:
				all_words_dict[word] = 1
	
	#Sort in reverse order according to the value of the key
	all_words_tuple_list = sorted(all_words_dict.items(), key = lambda f:f[1], reverse = True)
	all_words_list, all_words_nums = zip(*all_words_tuple_list)	
	all_words_list = list(all_words_list)						
	return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list

def MakeWordsSet(words_file):
	# Function Description: Read the contents of the file and de-duplicate them
	words_set = set()											
	with open(words_file, 'r', encoding = 'utf-8') as f:		
		for line in f.readlines():								
			word = line.strip()									
			if len(word) > 0:									
				words_set.add(word)								
	return words_set 											


def TextFeatures(train_data_list, test_data_list, feature_words):
	'''
	Function:Acodding to the feature_words and vector them

	Parameters:
		train_data_list - train set
		test_data_list - test set
		feature_words - feature set 
	Returns:
		train_feature_list - Vectorization List of Training Sets
		test_feature_list - Vectorization List of test Sets
	Modify:
		2018-12-22
	'''
	def text_features(text, feature_words):																		
		text_words = set(text)
		features = [1 if word in text_words else 0 for word in feature_words]
		return features
	train_feature_list = [text_features(text, feature_words) for text in train_data_list]
	test_feature_list = [text_features(text, feature_words) for text in test_data_list]
	return train_feature_list, test_feature_list				


def words_dict(all_words_list, deleteN, stopwords_set = set()):
	'''
	Function: text features select
	Parameters:
		all_words_list - List of all text in training set
		deleteN - Delete the most frequent deleteN words
		stopwords_set - remark the stop words
	Returns:
		feature_words - features
	Modify:
		2018-12-22
	'''
	feature_words = []							
	n = 1
	for t in range(deleteN, len(all_words_list), 1):
		if n > 1000:							
			break								

		if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
			feature_words.append(all_words_list[t])
		n += 1
	return feature_words


def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
	classifier = MultinomialNB().fit(train_feature_list, train_class_list)
	test_accuracy = classifier.score(test_feature_list, test_class_list)
	return test_accuracy

if __name__ == '__main__':
	#text preprocessing
	folder_path = './data/Sample'				
	all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)

	# generate stopwords_set
	stopwords_file = './stopwords_cn.txt'
	stopwords_set = MakeWordsSet(stopwords_file)


	test_accuracy_list = []
	deleteNs = range(0, 1000, 20)				#0 20 40 60 ... 980
	for deleteN in deleteNs:
		feature_words = words_dict(all_words_list, deleteN, stopwords_set)
		train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
		test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
		test_accuracy_list.append(test_accuracy)

	# ave = lambda c: sum(c) / len(c)
	# print(ave(test_accuracy_list))

	plt.figure()
	plt.plot(deleteNs, test_accuracy_list)
	plt.title('Relationship of deleteNs and test_accuracy')
	plt.xlabel('deleteNs')
	plt.ylabel('test_accuracy')
	plt.show()
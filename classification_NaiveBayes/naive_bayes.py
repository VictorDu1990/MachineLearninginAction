# encoding=utf-8

import pandas as pd
import numpy as np
import cv2
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

# binaryzation processing
def binaryzation(img):
    cv_img = img.astype(np.uint8)  
    cv2.threshold(cv_img, 50, 1, cv2.THRESH_BINARY_INV, cv_img)  
    return cv_img

# Training, calculating the prior probability and conditional probability
def Train(trainset, train_labels):
    prior_probability = np.zeros(class_num)                         # prior probability
    conditional_probability = np.zeros((class_num, feature_len, 2))   # conditional probability

    #  calculating
    for i in range(len(train_labels)):
        img = binaryzation(trainset[i])     # Picture binarization, so that each feature has only two values of 0 and 1
        label = train_labels[i]

        prior_probability[label] += 1

        for j in range(feature_len):
            conditional_probability[label][j][img[j]] += 1

    # Attribute conditional probability to [1,10001]
    for i in range(class_num):
        for j in range(feature_len):

            # After binarization, so that each feature has only two values of 0 and 1
            pix_0 = conditional_probability[i][j][0]
            pix_1 = conditional_probability[i][j][1]

            # calculating the 0,1 value conditional probability
            probalility_0 = (float(pix_0)/float(pix_0+pix_1))*10000 + 1
            probalility_1 = (float(pix_1)/float(pix_0+pix_1))*10000 + 1

            conditional_probability[i][j][0] = probalility_0
            conditional_probability[i][j][1] = probalility_1

    return prior_probability, conditional_probability

# calculating probability
def calculate_probability(img, label):
    probability = int(prior_probability[label])

    for j in range(feature_len):
        probability *= int(conditional_probability[label][j][img[j]])

    return probability

# predict
def Predict(testset, prior_probability, conditional_probability):
    predict = []

    # For each input x, the class with the greatest posteriori probability is used as the class output of X.
    for img in testset:

        img = binaryzation(img)  # binarization

        max_label = 0
        max_probability = calculate_probability(img, 0)

        for j in range(1, class_num):
            probability = calculate_probability(img, j)

            if max_probability < probability:
                max_label = j
                max_probability = probability

        predict.append(max_label)

    return np.array(predict)


class_num = 10  # MINST dataset have 10 kinds of labels，as:“0,1,2,3,4,5,6,7,8,9”
feature_len = 784  # MINST dataset image[28*28=784] features（pixels）

if __name__ == '__main__':
    print("Start read data")
    time_1 = time.time()

    raw_data = pd.read_csv('../data/train.csv', header=0)  # read csv dataset
    data = raw_data.values

    features = data[::, 1::]
    labels = data[::, 0]
    # To avoid over-fitting, cross-validation is used, and 33% data is randomly selected as test set and the rest is training set.
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=0)

    time_2 = time.time()
    print('read data cost %f seconds' % (time_2 - time_1))

    print('Start training')
    prior_probability, conditional_probability = Train(train_features, train_labels)
    time_3 = time.time()
    print('training cost %f seconds' % (time_3 - time_2))

    print('Start predicting')
    test_predict = Predict(test_features, prior_probability, conditional_probability)
    time_4 = time.time()
    print('predicting cost %f seconds' % (time_4 - time_3))

    score = accuracy_score(test_labels, test_predict)
    print("The accruacy score is %f" % score)

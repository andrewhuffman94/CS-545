# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 12:08:31 2020

@author: Andrew
"""

import pandas as pd
import numpy as np
import scipy.stats
import math



##### Read data, split into spam and non-spam, and partition into training and test data sets #####
data = pd.read_csv("spambase_data.csv")
mask = data["spam"]==1
spam = data[data["spam"]==1]
notspam = data[data["spam"]==0]
training_spam = spam[0:math.ceil(len(spam)/2)]
test_spam = spam[math.ceil(len(spam)/2):len(spam)]
training_notspam = notspam[0:math.ceil(len(notspam)/2)]
test_notspam = notspam[math.ceil(len(notspam)/2):len(notspam)]
training_data = pd.concat([training_spam,training_notspam],axis = 0)
test_data =pd.concat([test_spam,test_notspam],axis=0)
training = pd.DataFrame.to_numpy(training_data)
testing = pd.DataFrame.to_numpy(test_data)


##### Compute P(spam), P(notspam), mean and standard deviation for each feature given each class ####
spam_class = training[np.where(training[:,57]==1)]
notspam_class = training[np.where(training[:,57]==0)]
#P_spam = spam_class.shape[0]/training.shape[0]
#P_notspam = notspam_class.shape[0]/training.shape[0]
P_spam = 1
P_notspam = 1
mu_spam = np.mean(spam_class,axis=0)
stdev_spam = np.std(spam_class,axis=0)
stdev_spam = np.where(stdev_spam==0,0.0001,stdev_spam)
mu_notspam = np.mean(notspam_class,axis=0)
stdev_notspam = np.std(notspam_class,axis=0)
stdev_notspam = np.where(stdev_notspam==0,0.0001,stdev_notspam)


##### Calculate P(x|spam), P(x|notspam), and determine which class maximizes posterior probability  #####
correct_predictions = 0
confusion_matrix = np.zeros((2,2))
for i in range(0,testing.shape[0]):
    P_spam_class = np.zeros((1,testing.shape[1]-1))
    P_notspam_class = np.zeros((1,testing.shape[1]-1))
    x = testing[i,:].reshape(1,testing.shape[1])
    for j in range(0,testing.shape[1]-1):
        P_spam_class[0,j] = (scipy.stats.norm.pdf(x[0,j],mu_spam[j],stdev_spam[j]))
        P_notspam_class[0,j] = (scipy.stats.norm.pdf(x[0,j],mu_notspam[j],stdev_notspam[j]))
    logP_spam_class = np.log(P_spam_class)
    logP_notspam_class = np.log(P_notspam_class)
    spam_posterior = math.log(P_spam)+np.sum(logP_spam_class,axis=1)
    notspam_posterior = math.log(P_notspam)+np.sum(logP_notspam_class,axis=1)
    if spam_posterior>notspam_posterior:
        prediction = 1
    else:
        prediction = 0
    if prediction == testing[i][57]:
        correct_predictions = correct_predictions+1
    confusion_matrix[int(testing[i,57]),int(prediction)] = confusion_matrix[int(testing[i,57]),int(prediction)]+1


##### Analyze Naive Bayes predictor performance #####
accuracy = correct_predictions/testing.shape[0]
precision = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
recall = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[1,0])
     
        




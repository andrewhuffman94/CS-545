# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 20:43:07 2020

@author: Andrew
"""
import pandas as pd
import numpy as np
import scipy
from scipy import special
import matplotlib.pyplot as plt
import math
import time
        
def MLP(training,testing,inputs,hidden,outputs,eta,alpha,epochs):  
    global epoch
    global correct_training
    global correct_test
    global accuracy_training
    global accuracy_test
    ### Initialize arrays and variables and generate random weights
    correct_training = np.zeros((epochs,1))
    correct_test = np.zeros((epochs,1))
    accuracy_training = np.zeros((epochs,1))
    accuracy_test = np.zeros((epochs,1))
    epoch = np.zeros((epochs,1))
    W_h = ((np.random.rand(n,inputs+1))-0.5)/10  ### Generate inputs-to-hidden weights
    W_o = ((np.random.rand(outputs,n+1))-0.5)/10  ### Generate hidden-to-outputs weightss
    for e in range(0,epochs):
        
        ### Vary batch sizes so that I get "large" changes for early epochs (should be "far" from minimum) and "small" changes for later epochs (should be closer to minimum)
        if e < 35:
            batch_size = 2
        elif e>=35 & e<40:
            batch_size = 5
        else:
            batch_size = 15
        ### Store data as arrays
        training_data = training.to_numpy()
        test_data = testing.to_numpy()
        ### Permute training data rows
        training_data = np.random.permutation(training_data)
        ### Create arrays with inputs plus bias for training and testing data
        bias_training = np.ones((training_data.shape[0],1))
        bias_testing = np.ones((test_data.shape[0],1))
        t_training = training_data[:,0].reshape(training_data.shape[0],1)
        x = training_data[:,1:]/255
        x_training = np.concatenate((bias_training,x),axis=1) 
        t_testing = test_data[:,0].reshape(test_data.shape[0],1)
        x = test_data[:,1:]/255
        x_testing = np.concatenate((bias_testing,x),axis=1)
        ### Count epochs
        epoch[e,0] = (e+1)
        ### Initialize arrays and variable that need to reset with each epoch
        confusion = np.zeros((outputs,outputs))
        correct_predictions_training = 0
        correct_predictions_testing = 0
        dW_o_t = np.zeros((W_o.shape[0],W_o.shape[1])) 
        dW_h_t = np.zeros((W_h.shape[0],W_h.shape[1]))
        dW_o = np.zeros((W_o.shape[0],W_o.shape[1])) 
        dW_h = np.zeros((W_h.shape[0],W_h.shape[1]))
        
        ### Pick datum, pass through network, and check predicted output
        for i in range(0,training_data.shape[0]):
            t_k = np.ones((outputs,1))*0.1
            x_k = x_training[i].reshape(training_data.shape[1],1)
            true_label = t_training[i,0]
            t_k[true_label,0] = 0.9
            z_h = np.dot(W_h,x_k)
            h_n = (1/(1+np.exp(-z_h)))
            h = np.concatenate((np.ones((1,1)),h_n))
            z_o = np.dot(W_o,h)
            o_k = 1/(1+np.exp(-z_o))
            predicted_label=np.argmax(scipy.special.softmax(o_k))
            
            ### Implement backprop learning rule (o:output)
            if predicted_label != true_label:
                delta_o = np.zeros((outputs,1))
                delta_h = np.zeros(((n+1),1))
                delta_o = o_k*(1-o_k)*(t_k-o_k)
                delta_h = h*(1-h)*np.dot(np.transpose(W_o),delta_o)
                dW_o_t = eta*np.dot(delta_o,np.transpose(h))+dW_o_t
                dW_h_t = eta*np.dot(delta_h[1:,:],np.transpose(x_k))+dW_h_t
                if i%batch_size == 0:
                    W_o = W_o+((dW_o_t+alpha*dW_o)/batch_size)
                    W_h = W_h+((dW_h_t+alpha*dW_h)/batch_size)
                dW_o = dW_o_t
                dW_h = dW_h_t
                if i%batch_size == 0:
                    dW_o_t = np.zeros((W_o.shape[0],W_o.shape[1])) 
                    dW_h_t = np.zeros((W_h.shape[0],W_h.shape[1]))
            
            ### Count correct predictions
            elif predicted_label == true_label:
                correct_predictions_training = correct_predictions_training+1
            ### Calculate training accuracy per epoch
            if i == (training_data.shape[0]-1):
                correct_training[e] = correct_predictions_training
                accuracy_training[e] = (correct_predictions_training/60000)*100
        
        ### Pass test data through network and predict label
        for i in range(0,test_data.shape[0]):
            t_k = np.ones((outputs,1))*0.1
            x_k = x_testing[i].reshape(test_data.shape[1],1)
            true_label = t_testing[i,0]
            t_k[true_label,0] = 0.9
            z_h = np.dot(W_h,x_k)
            h_n = (1/(1+np.exp(-z_h)))
            h = np.concatenate((np.ones((1,1)),h_n))
            z_o = np.dot(W_o,h)
            o_k = 1/(1+np.exp(-z_o))
            predicted_label=np.argmax(scipy.special.softmax(o_k))
            ### Generate confusion matrix
            if e == epochs-1:
                confusion[true_label,predicted_label] = confusion[true_label,predicted_label]+1
            
            ### Cacluate and store accuracy per epoch
            if predicted_label == true_label:
                correct_predictions_testing = correct_predictions_testing+1
            if i == (test_data.shape[0]-1):
                correct_test[e] = correct_predictions_testing
                accuracy_test[e] = (correct_predictions_testing/10000)*100
    accuracy_training_df = pd.DataFrame(accuracy_training)
    correct_test_df = pd.DataFrame(correct_test)
    accuracy_test_df = pd.DataFrame(accuracy_test)
    output_df = pd.concat([correct_training_df,accuracy_training_df,correct_test_df,accuracy_test_df],axis=1)
    output_df.to_excel("PLA Summary" +str(eta) + str(n) +".xlsx")
    ### Save confusion matrix
    confusion_df = pd.DataFrame(confusion)
    confusion_df.to_excel("Confusion_Matrix"+str(eta)+  str(n)+".xlsx")
    
    ### Plot training and test accuracy vs epoch
    plt.figure(n)
    plt.plot(epoch,accuracy_training,"-b",label="Training")
    plt.plot(epoch,accuracy_test,"-r",label="Test")
    plt.title("eta ="+str(eta))
    plt.xlabel("Epoch")
    plt.xlim(0,50)
    plt.ylabel("Accuracy")
    plt.ylim(75,100)
    plt.legend(loc="lower left")

### Load training and test data sets
training = pd.read_csv("mnist_train.csv")
testing = pd.read_csv("mnist_test.csv")


### Define parameters for experiment 1 and call function
for a in range(2,3):
    if a == 0:
        eta = 0.2
        alpha = 0.9
        n = 20
        inputs = 784
        outputs = 10
        epochs = 50
        MLP(training,testing,inputs,n,outputs,eta,alpha,epochs)
    elif a == 1:
        eta = 0.2
        alpha = 0.9
        n = 50
        inputs = 784
        epochs = 50
        outputs = 10
        MLP(training,testing,inputs,n,outputs,eta,alpha,epochs)
    elif a == 2:
        eta = 0.2
        alpha = 0.9
        n = 100
        inputs = 784
        outputs = 10
        epochs = 50
        MLP(training,testing,inputs,n,outputs,eta,alpha,epochs)
## Define parameters for experiment 2 and call function
for a in range(0,3):
    if a == 0:
        eta = 0.2
        alpha = 0
        n = 100
        inputs = 784
        outputs = 10
        epochs = 50
        MLP(training,testing,inputs,n,outputs,eta,alpha,epochs)
    if a == 1:
        eta = 0.2
        alpha = 0.25
        n = 100
        inputs = 784
        outputs = 10
        epochs = 50
        MLP(training,testing,inputs,n,outputs,eta,alpha,epochs)
    if a == 2:
        eta = 0.2
        alpha = 0.50
        n = 100
        inputs = 784
        outputs = 10
        epochs = 50
        MLP(training,testing,inputs,n,outputs,eta,alpha,epochs)
        
### Select portions of data and call function for experiment 3     
eta = 0.2
alpha = 0.9
n = 100
inputs = 784
outputs = 10
epochs = 50
training_50 = training.iloc[::2,:] ### Select 50% of training data 
testing_50 = testing.iloc[::2,:] ### Select 50% of training data 
training_25 = training.iloc[::4,:] ### Select 25% of training data 
testing_25 = testing.iloc[::4,:] ### Select 25% of training data 
MLP(training_25,testing_25,inputs,n,outputs,eta,alpha,epochs)
MLP(training_50,testing_50,inputs,n,outputs,eta,alpha,epochs)
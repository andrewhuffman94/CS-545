# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 00:45:56 2020

@author: Andrew
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##### Read data #####
data = pd.read_csv("cluster_dataset.txt",sep="\s+",header=None)
X = pd.DataFrame(data)


def k_means(X,k,r):
    x1 = pd.DataFrame.to_numpy(X[0])
    x2 = pd.DataFrame.to_numpy(X[1])
    mu = np.mean(X)
    stdev = np.std(X)
    SSW = np.zeros((r,1))
    Assignments = np.zeros((X.shape[0],r))
    Classified_Data = np.zeros((X.shape[0],2*k*r))
    for t in range(0,r):
        error = 0
        centroids = np.random.normal(loc=mu,scale=stdev,size=(k,X.shape[1])) ##### Initialize Centroids #####
        delta = 1
        old_classes = np.ones((X.shape[0],1))*100
        classified_data = []
        colors = ["r","b","g","c","m","y","k"]
        iteration = 0
        while delta>0:
            iteration = iteration+1
            new_centroids = []
            ###### Assign Classes #####
            D = []
            for i in range(0,k):
                D.append(np.linalg.norm((X-centroids[i][:]),ord=2,axis=1))
            d = np.asarray(D)
            classes = np.argmin(d,axis=0)
            delta = np.sum((classes-old_classes)**2) ##### Compute change in assignments #####
            
            ##### Update centroids #####
            check = []
            for c in range(0,k):
                 check.append(np.array(classes==c,dtype=int))
            mask = np.asarray(check)
            for c in range(0,k):
                a = np.multiply(mask[c],x1).reshape(X.shape[0],1)
                b = np.multiply(mask[c],x2).reshape(X.shape[0],1)
                ab = np.concatenate((a,b),axis=1)
                class_data1 = x1[np.nonzero(a)[0]]
                class_data2 = x2[np.nonzero(b)[0]]
                Classified_Data[0:class_data1.shape[0],((2*c)+(2*c*t))] = class_data1
                Classified_Data[0:class_data2.shape[0],(1+(2*c)+(2*c*t))] = class_data2
                new_centroids.append(np.mean(ab,axis=0))
                plot_centroids = np.asarray(new_centroids)
                error = error+np.linalg.norm(ab[np.all(ab!=0,axis=1)]-np.mean(ab,axis=0),ord=2) ##### Update SSW #####
                lab = "Class "+str(c+1)
                clab = "Class "+str(c+1)+" "+"Centroid"
                csym = ["*","D","s","*","D","s"]
                csize = [50,20,30,50,20,30]
                filename = "Clustering_t="+str(t)+"_iteration="+str(iteration)
                plt.figure(t)
                plt.scatter(class_data1,class_data2,s=5,c=colors[c],marker=".",label=lab)
                plt.scatter(plot_centroids[c,0],plot_centroids[c,1],s=csize[c],c="k",marker=csym[c],label=clab)
                plt.xlabel = "x1"
                plt.ylabel = "x2"
                plt.legend()
                plt.savefig(filename)
            plt.show()
            centroids = np.asarray(new_centroids)
            Assignments[:,t] = classes.reshape(X.shape[0],1)[:,0]
            old_classes = classes
            SSW[t] = error
            plt.figure(t)
    pd.DataFrame(Assignments).to_excel("Class Assignments.xlsx")
    pd.DataFrame(SSW).to_excel("SSW.xlsx")    
 
    
def fuzzy_c_means(X,k,r,m):
    global SSW
    x1 = pd.DataFrame.to_numpy(X[0]).reshape(X.shape[0],1)
    x2 = pd.DataFrame.to_numpy(X[1]).reshape(X.shape[0],1)
    SSW = np.zeros((r,1))
    Assignments = np.zeros((X.shape[0],r))
    Classified_Data = np.zeros((X.shape[0],2*k*r))
    for t in range(0,r):
        error = 0
        delta = 1
        old_classes = np.ones((X.shape[0],1))*100
        classified_data = []
        colors = ["r","b","g","c","m","y","k"]
        iteration = 0
        
        ##### Initialize Membership Weights #####
        matrix = np.random.rand(X.shape[0],k)
        W = (matrix/matrix.sum(axis=1).reshape(X.shape[0],1))
        
        while delta>0:
            iteration = iteration+1
            
            #### Update centroids #####
            check = []
            w_sum = np.sum(W,axis=0)
            new_centroids = []
            for c in range(0,k):
                W_v = np.power(W[:,c].reshape(X.shape[0],1),m)
                num = np.sum(np.multiply(X,W_v),axis=0)
                denom = np.sum(W_v,axis=0)
                centroid = num/denom
                new_centroids.append(centroid)
                
            ##### Assign classes based on membership grades and generate plots######
            classes = np.argmax(W,axis=1)
            delta = np.sum((classes-old_classes)**2)
            print(delta)
            for c in range(0,k):
                 check.append(np.array(classes==c,dtype=int))
            mask = np.transpose(np.asarray(check))
            for c in range(0,k):
                a = np.multiply(mask[:,c].reshape(X.shape[0],1),x1).reshape(X.shape[0],1)
                b = np.multiply(mask[:,c].reshape(X.shape[0],1),x2).reshape(X.shape[0],1)
                ab = np.concatenate((a,b),axis=1)
                class_data1 = x1[np.nonzero(a)[0]]
                class_data2 = x2[np.nonzero(b)[0]]
                Classified_Data[0:class_data1.shape[0],((2*c)+(2*c*t))] = class_data1[:,0]
                Classified_Data[0:class_data2.shape[0],(1+(2*c)+(2*c*t))] = class_data2[:,0]
                plot_centroids = np.asarray(new_centroids).reshape(k,2)
                error = error+np.linalg.norm(ab[np.all(ab!=0,axis=1)]-np.mean(ab,axis=0),ord=2) ##### Update SSW #####
                lab = "Class "+str(c+1)
                clab = "Class "+str(c+1)+" "+"Centroid"
                csym = ["*","D","s","*","D","s"]
                csize = [50,20,30,50,20,30]
                filename = "Clustering_t="+str(t)+"_iteration="+str(iteration)
                plt.figure(t)
                plt.scatter(class_data1,class_data2,s=5,c=colors[c],marker=".",label=lab)
                plt.scatter(plot_centroids[c,0],plot_centroids[c,1],s=csize[c],c="k",marker=csym[c],label=clab)
                plt.xlabel = "x1"
                plt.ylabel = "x2"
                plt.legend()
                plt.savefig(filename)
            plt.show()
            centroids = plot_centroids 
            Assignments[:,t] = classes.reshape(X.shape[0],1)[:,0]
            old_classes = classes
            SSW[t] = error
            
           ###### Update membership grades #####
            d = []
            w = []
            
            for c in range(0,k):
                dif = X-centroids[c,:]
                norm2 = np.linalg.norm(dif,ord=2,axis=1)                
                d.append(norm2)
            D = np.transpose(np.asarray(d))
            D = np.power(D,(1/(m-1)))
            for c in range(0,k):
                d_v = D[:,c]
                A = 1/d_v
                w_num = 1/d_v
                w_denom = np.sum((1/D),axis=1)
                w.append(w_num/w_denom)
            W = np.transpose(np.asarray(w))      
            
    pd.DataFrame(Assignments).to_excel("fuzzy cmeans Class Assignments.xlsx")
    pd.DataFrame(SSW).to_excel("fuzzy cmeans SSW.xlsx")    

    
    
k_means(X,3,10)
fuzzy_c_means(X,3,10,2)


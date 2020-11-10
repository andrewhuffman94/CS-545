# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 12:47:16 2020

@author: Andrew
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:59:13 2020

@author: Andrew
"""
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import norm
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import math
import time

start_time = time.time()
##### Load and preprocess dataset #####

data = pd.read_csv("IMDB Dataset.csv")
text = data["review"]
labels = data["sentiment"]
priors = np.ones((1,2))*0.5
X = text
y = labels
x_train, x_test, y_train, y_test = train_test_split(X, y,)
vectorizer_count = CountVectorizer(stop_words="english",max_features=2000)
X_train = vectorizer_count.fit_transform(x_train)
X_train_dense = X_train.toarray()
X_test = vectorizer_count.transform(x_test)
X_test_dense = X_test.toarray()
mnb = MultinomialNB()
y_pred = mnb.fit(X_train_dense, y_train).predict(X_test_dense)
incorrect_predictions_count = (y_pred != y_test).sum()
accuracy_count = 100*(1-(incorrect_predictions_count/y_pred.shape[0]))

vectorizer_tfidf = TfidfVectorizer(stop_words="english",ngram_range=(1,1))
X_train = vectorizer_tfidf.fit_transform(x_train)
X_test = vectorizer_tfidf.transform(x_test)
mnb = MultinomialNB(alpha=0.5)
y_pred_mnb = mnb.fit(X_train,y_train).predict(X_test)
incorrect_predictions_tfidf = (y_pred != y_test).sum()
accuracy_tfidf = 100*(1-(incorrect_predictions_tfidf/y_pred.shape[0]))


run_time = time.time()-start_time









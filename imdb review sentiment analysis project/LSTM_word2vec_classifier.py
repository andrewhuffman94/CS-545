# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:07:54 2020

@author: Andrew
"""
# This code is based on Alex Cherniuk's Kaggle notebook published at https://www.kaggle.com/alexcherniuk/imdb-review-word2vec-bilstm-99-acc
import warnings
warnings.filterwarnings('ignore')

# Modules for data manipulation
import numpy as np
import pandas as pd
import re

# Tools for preprocessing input data
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Tools for creating ngrams and vectorizing input data
from gensim.models import Word2Vec, Phrases

# Tools for building a model
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences

def clean_review(raw_review: str) -> str:
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review, "lxml").get_text()
    # 2. Remove non-letters
    letters_only = REPLACE_WITH_SPACE.sub(" ", review_text)
    # 3. Convert to lower case
    lowercase_letters = letters_only.lower()
    return lowercase_letters


def lemmatize(tokens: list) -> list:
    # 1. Lemmatize
    tokens = list(map(lemmatizer.lemmatize, tokens))
    lemmatized_tokens = list(map(lambda x: lemmatizer.lemmatize(x, "v"), tokens))
    # 2. Remove stop words
    meaningful_words = list(filter(lambda x: not x in stop_words, lemmatized_tokens))
    return meaningful_words


def preprocess(review: str, total: int, show_progress: bool = True) -> list:
    if show_progress:
        global counter
        counter += 1
        print('Processing... %6i/%6i'% (counter, total), end='\r')
    # 1. Clean text
    review = clean_review(review)
    # 2. Split into individual words
    tokens = word_tokenize(review)
    # 3. Lemmatize
    lemmas = lemmatize(tokens)
    # 4. Join the words back into one string separated by space,
    # and return the result.
    return lemmas

data = pd.read_csv("IMDB Dataset.csv")
all_reviews = data["review"]
labels =  data["sentiment"].isin(["positive"]).to_numpy().astype(int)
counter = 0
REPLACE_WITH_SPACE = re.compile(r'[^A-Za-z\s]')
stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()

all_reviews = np.array(list(map(lambda x: preprocess(x, len(all_reviews)), all_reviews)))

X_train_data = all_reviews


bigrams = Phrases(sentences=all_reviews)
trigrams = Phrases(sentences=bigrams[all_reviews])
embedding_vector_size = 256
trigrams_model = Word2Vec(sentences = trigrams[bigrams[all_reviews]],size = embedding_vector_size,min_count=3, window=5, workers=4)
def vectorize_data(data, vocab: dict) -> list:
    print('Vectorize sentences...', end='\r')
    keys = list(vocab.keys())
    filter_unknown = lambda word: vocab.get(word, None) is not None
    encode = lambda review: list(map(keys.index, filter(filter_unknown, review)))
    vectorized = list(map(encode, data))
    print('Vectorize sentences... (done)')
    return vectorized

print('Convert sentences to sentences with ngrams...', end='\r')
X_data = trigrams[bigrams[X_train_data]]
print('Convert sentences to sentences with ngrams... (done)')
input_length = 200
X_pad = pad_sequences(sequences=vectorize_data(X_data, vocab=trigrams_model.wv.vocab),maxlen=input_length,padding='post')
print('Transform sentences to sequences... (done)')
X_train, X_test, y_train, y_test = train_test_split(X_pad,labels,test_size=0.05,shuffle=True,random_state=42)


def build_model(embedding_matrix: np.ndarray, input_length: int):
    model = Sequential()
    model.add(Embedding(input_dim = embedding_matrix.shape[0],output_dim = embedding_matrix.shape[1], input_length = input_length,weights = [embedding_matrix],trainable=False))
    model.add(Bidirectional(LSTM(128, recurrent_dropout=0.1)))
    model.add(Dropout(0.20))
    model.add(Dense(64))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    return model

model = build_model(embedding_matrix=trigrams_model.wv.vectors,input_length=input_length)

model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

history = model.fit(x=X_train,y=y_train,validation_data=(X_test, y_test),batch_size=100,epochs=50)

y_train_pred = model.predict_classes(X_train)
y_test_pred = model.predict_classes(X_test)





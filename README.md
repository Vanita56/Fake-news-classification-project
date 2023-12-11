# Fake-news-classification-project
The aim of this project is to build a machine learning model capable of predicting whether a given news article is fake or genuine.
I took dataset from kaggle, it was a large daatset.
Dataset: news articals as real and fake.
https://www.kaggle.com/c/fake-news/data?select=train.csvidtitleauthortextlabel-marks  
The dataset is divided into 5 columns:
1. id
2. Title
3. Author
4. Text
5. Label (Fake is represented by '1' and real by '0')
Search for null values in dataset using isnull() function and replaced with space(.fillna(' ')) as it was a large dataset containing (20800 cols, 5 rows)
I combine title and author column in one new column named content.
X Y->label
preprocessing ->>
1 stemming (port_stem = PorterStemmer())
2  TF-IDF vectorizer = TfidfVectorizer()  #Term Frequency and Inverse Doc Frequency (CONVERTED TEXT INTO NUMBERS)
   vectorizer.fit(X)
   X = vectorizer.transform(X)
SPLITING DATASET into training and testing dataset (random_size=.2) 80 20%
Build the model and then test it using predict method and find the accuracy score to evaluate the performance of the mmodel
Used LINEAR KERNEL IN SVM 
In this project I build two models to evaluate the performance of our project. SVM has more Accuracy score than logistic regression.
we got 97.9% accuracy score in logistic regression and 99.1% accuracy score in SVM.

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

//LogisticRegression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)

//SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)

from sklearn.metrics import accuracy_score
classifier.fit(X_train, Y_train)

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 14:27:36 2021

@author: Salwa
"""

import numpy as np 
#sklearn built-in dataset 
from sklearn.datasets import load_breast_cancer
#load the data 
data = load_breast_cancer()

#print(data)
#classification process begin 

from sklearn.model_selection import train_test_split
#split the data to train and test sets
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)

#import the classifier KNN

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
#train the model 
model.fit(x_train, y_train)
# calculate scores 
score_train= model.score(x_train, y_train)
score_test= model.score(x_test, y_test)
#predict the model
predictions = model.predict(x_test)

print(score_train)
print(score_test)
print(predictions)
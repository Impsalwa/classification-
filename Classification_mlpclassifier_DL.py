# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 10:41:02 2021

@author: Salwa
"""
#need to know what is future 
from __future__ import print_function, division
from future.utils import iteritems 
from builtins import range, input 
import numpy as np 
#sklearn built-in dataset 
from sklearn.datasets import load_breast_cancer
#load the data 
data = load_breast_cancer()
#using deep learning 
from sklearn.neural_network import MLPClassifier
#import scaler 
from sklearn.model_selection import train_test_split
#split the data to train and test sets
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)

from sklearn.preprocessing import StandardScaler
#fit anf trasfom scaler
scaler = StandardScaler()
x_train2= scaler.fit_transform(x_train)
x_test2 = scaler.transform(x_test)
#build the model 
model = MLPClassifier(max_iter= 500)
model.fit(x_train2, y_train)
#calculate scores 
score_train= model.score(x_train2, y_train)
score_test =model.score(x_test2, y_test)

#prediction
predictions = model.predict(x_test2)

print(score_train)
print(score_test)
print(predictions)


# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 17:29:00 2021

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

#check the type of data
type(data)
#to inderstand data and their features 
data.keys()

data.data.shape
data.target
data.target_names
data.target.shape
data.feature_names

#classification process begin 

from sklearn.model_selection import train_test_split
#split the data to train and test sets
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)

#import the classifier 
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
#train the model 
model.fit(x_train, y_train)
# calculate scores 
score_train= model.score(x_train, y_train)
score_test= model.score(x_test, y_test)
#predict the model
predictions = model.predict(x_test)
#see what the predicts are 
predictions 

#manualy calculate the score of your model 
N= len(y_test)
score_test2= np.sum(predictions == y_test)/ N #can call np.mean() 

print(score_train)
print(score_test)
print(predictions)





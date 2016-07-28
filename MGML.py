# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:48:46 2016

@author: nea040
"""

from sklearn.ensemble import RandomForestClassifier
import numpy as np

# import data as numpy arrays

train_file = open("./data/MGML_train.txt", "rb")
test_file = open("./data/MGML_test.txt", "rb")

def create_numpy_array(data_file):
    test_answers = []
    data = []
    for line in data_file:
        line = line.strip()
        cols = line.split("\t")
        test_answers.append(cols[1])
        data.append(cols)
    return np.array(data), test_answers

train_array = create_numpy_array(train_file)[0]
test_array, test_answers = create_numpy_array(test_file)

# create random forest object

forest = RandomForestClassifier(n_estimators = 100)

# fit training data

#print train_array[0::,2::]

forest = forest.fit(train_array[0::,2::],train_array[0::,1])

output = forest.predict(test_array[0::,2::])

# check result against known answers

correct = 0
incorrect = 0
for result, answer in zip(output, test_answers):
    if result == "1" or answer == "1": 
        if answer == result:
            correct += 1
        else:
            incorrect += 1

print "test accuracy:", correct, incorrect, (float(correct) / float(correct + incorrect))
    
    
    
    
    
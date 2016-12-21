#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
features_train = features_train[:len(features_train)] 
labels_train = labels_train[:len(labels_train)] 
from sklearn.svm import SVC
classifier = SVC(kernel="rbf", C=10000)
classifier.fit(features_train, labels_train)
pred = classifier.predict(features_test)
chris = 0
map = {}
for i in pred:
    if i not in map:
        map[i] = 0
    map[i] += 1
print map
# print "training time:", round(time()-t0, 3), "s" #179.374s
# print classifier.score(features_test, labels_test) #0.984072810011
# print "testinging time:", round(time()-t0, 3), "s" #20.126s
#########################################################



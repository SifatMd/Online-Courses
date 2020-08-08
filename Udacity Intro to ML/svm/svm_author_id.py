#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
import numpy as np 
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
#from thundersvm import SVC #this is for applying multicore SVC (only CPU)


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

# reducing training size
# features_train = features_train[: len(features_train)//100]
# labels_train = labels_train[: len(labels_train)//100]

model = SVC(kernel='rbf', C=10000, gamma='auto')
t0 = time()
model.fit(features_train, labels_train)
print('training time: {}'.format(round(time()-t0, 3)))

t1 = time()
pred = model.predict(features_test)
print('prediction time: {}'.format(round(time()-t1, 3)))

test_acc = accuracy_score(pred, labels_test)
print('testing accuracy: {}'.format(test_acc))

temparr = [pred[10], pred[26], pred[50]]
temparr = ['Sara' if x==0 else "Chris" for x in temparr]
#print(temparr)

num_chris = np.count_nonzero(pred==1)
print("Prediction Count for Chris' email {}".format(num_chris))

#########################################################



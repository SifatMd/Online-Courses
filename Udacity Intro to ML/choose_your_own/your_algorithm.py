#!/usr/bin/python

from time import time 
import numpy as np 
import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(bumpy_slow, grade_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
#plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

# Naive Bayes
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# t0 = time()
# clf.fit(features_train, labels_train)
# print('training time: {}'.format(round(time()-t0, 4)))
# pred = clf.predict(features_test)
# test_acc = accuracy_score(labels_test, pred)
# print('Naive bayes testing acc {}'.format(test_acc))


# SVM
from sklearn.svm import SVC
#from thundersvm import SVC
clf = SVC(kernel='rbf', C=200)
t0 = time()
clf.fit(features_train, labels_train)
print('training time: {}'.format(round(time()-t0, 4)))
pred = clf.predict(features_test)
test_acc = accuracy_score(labels_test, pred)
print('SVM testing acc {}'.format(test_acc))


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split=20)
t0 = time()
clf.fit(features_train, labels_train)
print('training time: {}'.format(round(time()-t0, 4)))
pred = clf.predict(features_test)
test_acc = accuracy_score(labels_test, pred)
print('SVM testing acc {}'.format(test_acc))




# KNN
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
t0 = time()
clf.fit(features_train, labels_train)
print('training time: {}'.format(round(time()-t0, 4)))
pred = clf.predict(features_test)
test_acc = accuracy_score(labels_test, pred)
print('KNN testing acc {}'.format(test_acc))


# Adaboost
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=50)
t0 = time()
clf.fit(features_train, labels_train)
print('training time: {}'.format(round(time()-t0, 4)))
pred = clf.predict(features_test)
test_acc = accuracy_score(labels_test, pred)
print('Adaboost testing acc {}'.format(test_acc))


# Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50, n_jobs=1)
t0 = time()
clf.fit(features_train, labels_train)
print('training time: {}'.format(round(time()-t0, 4)))
pred = clf.predict(features_test)
test_acc = accuracy_score(labels_test, pred)
print('RandomForest testing acc {}'.format(test_acc))




# XGBoost
import xgboost as xgb
xgb_model = xgb.XGBClassifier(n_estimators=100)
t0 = time()
xgb_model.fit(np.array(features_train), np.array(labels_train))
print('training time: {}'.format(round(time()-t0, 4)))
pred = xgb_model.predict(np.array(features_test))
test_acc = accuracy_score(labels_test, pred)
print('XGB testing acc {}'.format(test_acc))





try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass


print('done')


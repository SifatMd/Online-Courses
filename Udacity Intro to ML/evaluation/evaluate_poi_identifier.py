#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier()
# clf.fit(features, labels)
#print('naive training accuracy: %.4f' % (clf.score(features, labels)))

# after splitting
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
print('Updated accuracy: %f' % (clf.score(features_test, labels_test)))


pred = clf.predict(features_test)

labels_test = [int(x) for x in labels_test]
pred = [int(x) for x in pred]

predicted_poi = 0
for i in pred:
    if i == 1:
        predicted_poi += 1
print('predicted pois: {}'.format(predicted_poi))

print('total people in test set: {}'.format(len(labels_test)))

# print(pred)
# print(labels_test)

#precision and recall
from sklearn.metrics import precision_score, recall_score
print('precision: {}'.format(precision_score(labels_test, pred, average='binary')))
print('recall: {}'.format(recall_score(labels_test, pred, average='binary')))


#custom data
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

true_positives = 0
for i, j in zip(predictions, true_labels):
    if i ==j and i == 1:
        true_positives += 1

print(true_positives)

true_negatives = 0
for i, j in zip(predictions, true_labels):
    if i ==j and i == 0:
        true_negatives += 1

print(true_negatives)

false_positives = 0
for i, j in zip(predictions, true_labels):
    if i==1 and j==0:
        false_positives += 1

print(false_positives)

false_negatives = 0
for i, j in zip(predictions, true_labels):
    if i ==0 and j == 1:
        false_negatives += 1

print(false_negatives)

print('precision: {}'.format(precision_score(true_labels, predictions)))
print('recall: {}'.format(recall_score(true_labels, predictions)))




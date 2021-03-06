#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import numpy as np

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'exercised_stock_options', \
        'bonus', 'expenses', 'long_term_incentive', 'total_stock_value', 'shared_receipt_with_poi', \
            'bonus_salary_ratio', 'total_stock_total_payment_ratio'] # You will need to use more features


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)


# flag = 0
# for key, value in data_dict.items():
#     print(key)
#     print(value)
#     flag += 1
#     if flag == 5:
#         break

# number of poi's
# num_pois = 0
# for key, val in data_dict.items():
#     if val['poi'] == True:
#         num_pois += 1
# print('number of poi in the dataset: {}'.format(num_pois))


### Task 2: Remove outliers
data_dict.pop('TOTAL') # removing TOTAL key from data_dict as it is not anyone's record

print('total legit data points: %d' % (len(data_dict)))

### Task 3: Create new feature(s)
from sklearn.preprocessing import MinMaxScaler
minmaxscaler = MinMaxScaler()

bonus = []
salary = []
total_stock_value = []
total_payments = []
for key, value in data_dict.items():
    if value['bonus']=='NaN' or value['salary']=='NaN' or value['total_stock_value']=='NaN' or value['total_payments']=='NaN':
        bonus.append(0.)
        salary.append(0.)
        total_stock_value.append(0.)
        total_payments.append(0.)
    else:
        bonus.append(value['bonus'])
        salary.append(value['salary'])
        total_stock_value.append(value['total_stock_value'])
        total_payments.append(value['total_payments'])

bonus = np.array(bonus).reshape((-1, 1))
salary = np.array(salary).reshape((-1, 1))
total_stock_value = np.array(total_stock_value).reshape((-1, 1))
total_payments = np.array(total_payments).reshape((-1, 1))

#rescale values between 0 and 1
bonus_minmax = minmaxscaler.fit_transform(bonus)
salary_minmax = minmaxscaler.fit_transform(salary)
total_stock_value_minmax = minmaxscaler.fit_transform(total_stock_value)
total_payments_minmax = minmaxscaler.fit_transform(total_payments)

# print(len(data_dict), len(bonus_minmax))

#add new features to dictionary
i = 0
for key, value in data_dict.items():
    if bonus_minmax[i][0] == 0.:
        value['bonus_salary_ratio'] = 0.
        value['total_stock_total_payment_ratio'] = 0. 
    else:
        value['bonus_salary_ratio'] = bonus_minmax[i][0] / salary_minmax[i][0]
        value['total_stock_total_payment_ratio'] = total_stock_value_minmax[i][0] / total_payments_minmax[i][0]
    i += 1


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA 

clf = AdaBoostClassifier(n_estimators=100, learning_rate=.8)
estimators = [('reduce_dim', PCA(n_components=5)), ('clf', clf)]
pipe = Pipeline(estimators)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.25, random_state=42)


#train and validate
def train_and_validate(clf, features_train, features_test, labels_train, labels_test):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    #print(labels_test)
    accuracy = accuracy_score(labels_test, pred)
    precision = precision_score(labels_test, pred)
    recall = recall_score(labels_test, pred)
    f1score = f1_score(labels_test, pred)
    print('Classifier: {}'.format(clf))
    print('Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}'.format(accuracy, precision, recall, f1score))
    print()


train_and_validate(pipe, features_train, features_test, labels_train, labels_test)

### BEST RESULTS
# Classifier: Pipeline(steps=[('reduce_dim', PCA(n_components=4)), ('clf', GaussianNB())])
# Accuracy: 0.9722, Precision: 1.0000, Recall: 0.7500, F1-Score: 0.8571

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

# dump_classifier_and_data(clf, my_dataset, features_list)
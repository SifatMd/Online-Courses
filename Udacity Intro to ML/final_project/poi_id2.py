import sys 
import pickle 

sys.path.append('../tools/')

import numpy as np 
from tester import dump_classifier_and_data 
from feature_format import featureFormat, targetFeatureSplit 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import average_precision_score as precision_score 
from sklearn.metrics import recall_score, accuracy_score


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)


### Dataset Exploration
def datasetExploration(data_dict):
    print('Exploration of Enron E+F Dataset')
    keys = list(data_dict.keys())
    print(f'Number of data points: {len(keys)}')
    print('Number of features per person: {}'.format( len(data_dict[keys[0]].keys() )))
    num_poi = 0
    num_real_salary = 0
    for key, value in data_dict.items():
        if value['poi'] == True:
            num_poi += 1 
        if value['salary'] != "NaN":
            num_real_salary += 1

    print('Number of POIs: {}'.format(num_poi))
    print('Number of Non-POIs: {}'.format(len(keys)-num_poi))
    print('Number of people with quantified salary: {}'.format(num_real_salary))

def featuresMissingPercentage(data_dict):
    entries = list(data_dict.values())
    total = len(entries) 
    missingCounts = {}
    entries_keys = list(entries[0].keys())
    for key in entries_keys:
        missingCounts[key] = 0

    for value in entries:
        for key, val in value.items():
            if val == 'NaN':
                missingCounts[key] += 1

    missingPercentage = {}
    missingValues = {}

    for key,val in missingCounts.items():
        percent = val/ float(total)
        missingValues[key] = val #for a specific feature name, how many instances are missing
        missingPercentage[key] = percent #for a specific feature name, what % of instances are missing 
    
    print()
    return missingPercentage, missingValues


def displayMissingFeatures(missingPercentage, missingValues):
    for k, v in missingPercentage.items():
        print('{} of \'{}\' is missing; missing percentage is : {:.2f}'.format(missingValues[k], k, v))
    print()


def removeFeaturesWithMissingValuesAboveXpercentage(allFeatures, missingPercentage, threshold):
    newList = []
    for value in allFeatures:
        if value!='email_address' and float(missingPercentage[value])<=threshold: #only keep features which have less missing values
            newList.append(value) 
    print()
    return newList


# This function removes data point for which more than x% eg 80% of features are missing
def suspectDataPointForWhichMoreThanXpercentFeaturesIsNaN(data_dict, threshold):
    keys = list(data_dict.keys())
    num_features = len(data_dict[keys[0]].keys())
    data_dict_new = data_dict.copy()

    dataKeys = []
    for name, item in data_dict.items():
        num_nan = 0
        for key, value in item.items():
            if value == "NaN":
                num_nan += 1 
        
        if num_features == 0:
            nanPercent = 0
        else:
            nanPercent = num_nan/float(num_features)
        if nanPercent >= threshold:
            dataKeys.append(name)
            del data_dict_new[name] 

    return dataKeys, data_dict_new #remove data points(names of employees) for whom >threshold of feature values are missing



### Functions for removing outliers
# Gives you a visual view of existence of outliers
def PlotOutlier(data_dict, feature_x, feature_y):
    data = featureFormat(data_dict, [feature_x, feature_y, 'poi'])
    for point in data:
        x = point[0]
        y = point[1]
        poi = point[2]
        if poi:
            color='red'
        else:
            color='blue'
        plt.scatter(x, y, color=color)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show() 


def getFormattedFeaturesAndNameList(data_dict, features):
    featureList = {}
    nameList = {}
    for feature in features:
        featureList[feature] = []
        nameList[feature] = []
    
    for feature in features:
        for name, value in data_dict.items():
            for key, val in value.items(): #feature name and its respective value 
                if key in features and val != "NaN":
                    featureList[key].append(val)
                    nameList[key].append(name) #name is the name of the person whose features are in question
    
    return featureList, nameList # contains the legit feature values and corresponding names for each possible feature


def findSuspectedOutliers(featureList):
    suspectedOutliers = {}
    for key in featureList.keys():
        suspectedOutliers[key] = set() 

    for key, value in featureList.items():
        q1 = np.percentile(value, 25)
        q3 = np.percentile(value, 75)
        iqr = q3 - q1 
        floor = q1 - 1.5*iqr 
        ceiling = q3 + 1.5*iqr 

        for x in value:
            if ((x<floor) | (x>ceiling)):
                suspectedOutliers[key].add(x)
    
    return suspectedOutliers #contains the suspected outlier values for each feature


def findOutliers(data_dict, outlyingValues, threshold):
    outliers = {}
    outlyingKeys = outlyingValues.keys()

    for key in data_dict.keys():
        outliers[key] = 0 #key are names of each person in data_dict
    
    for key, value in data_dict.items():
        for k, val in value.items():
            if ((k in outlyingKeys) and (val in outlyingValues[k])):
                outliers[key] += 1 #key named person contains how many suspected outlier values

    filteredOutliers = {}
    total = 0
    for k, v in outliers.items():
        if v>0:
            filteredOutliers[k] = v #only contain names of persons with outlier values 
            total += v 

    realOutliers = []
    for key, value in filteredOutliers.items():
        if value/float(total) >= threshold:
            realOutliers.append(key) #contains those names of people for whom there are a lot of outlier values
    
    return realOutliers


def removeNaNvalues(my_dataset, financial_features):
    entries = list(my_dataset.values())
    feature_dict = {}
    for key in financial_features:
        feature_dict[key] = []

    for feature, val in entries[0].items():
        if val != "NaN" and val!=0 and val!=0. and feature in financial_features:
            feature_dict[feature].append(float(val))

    medians = {}
    for feature, values in feature_dict.items():
        if len(values) > 0:
            val = np.percentile(values, 50)
            medians[feature] = val 

    for name, entry in my_dataset.items():
        for feature, value in entry.items():
            if (value == 'NaN' or value==0. or value==0) and feature in list(medians.keys()):
                my_dataset[name][feature] = medians[feature]

    return my_dataset


### Create New Features
# from poi_to_this_person_ratio and from_this_person_to_poi_ratio
def computeRatio(messages, allMessages):
    ratio = 0
    if messages == "NaN" or allMessages == "NaN":
        return ratio 
    elif allMessages == 0:
        return ratio 
    
    ratio = messages / float(allMessages)
    return ratio 

def createNewFeatures(my_dataset):
    for poi_name in my_dataset:
        data_point = my_dataset[poi_name] 
        data_point['from_poi_to_this_person_ratio'] = computeRatio(data_point['from_poi_to_this_person'],
                data_point['to_messages'])
        data_point['from_this_person_to_poi_ratio'] = computeRatio(data_point['from_this_person_to_poi'], 
                data_point['from_messages'])
        
    return my_dataset, ['from_poi_to_this_person_ratio', 'from_this_person_to_poi_ratio']



#### ----------- Feature Selection Method -----------------------
def findKbestFeatures(data_dict, features_list, k):
    from sklearn.feature_selection import f_classif
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)
    try:
        k_best = SelectKBest(f_classif, k=k)
        k_best.fit(features, labels)
    except:
        print('exception occured')

    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    
    k_best_features = dict(sorted_pairs[:k])
    return k_best_features

features_list = ['poi']
email_features = ['email_address', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',\
                  'shared_receipt_with_poi']
email_features.remove('email_address')
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',\
                      'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',\
                      'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

# Combine all features
all_features = email_features + financial_features
all_features.remove('from_poi_to_this_person')
all_features.remove('from_messages')
all_features.remove('from_this_person_to_poi')
all_features.remove('other')

datasetExploration(data_dict)

missingPercentage, missingValues = featuresMissingPercentage(data_dict)

displayMissingFeatures(missingPercentage, missingValues)

#remove features for whom more than x%=70% values are missing
features_list = features_list + removeFeaturesWithMissingValuesAboveXpercentage(all_features, missingPercentage, 0.7)
all_features = removeFeaturesWithMissingValuesAboveXpercentage(all_features, missingPercentage, 0.8)

#### Remove Outliers
PlotOutlier(data_dict, 'total_payments', 'total_stock_value')

# Detect Outliers
featureList, nameList = getFormattedFeaturesAndNameList(data_dict, all_features)
outlyingValues = findSuspectedOutliers(featureList)
# print(outlyingValues)

# detected outliers are ['TOTAL']
detectedOutliers = findOutliers(data_dict, outlyingValues, 0.08) #returns list of names of people for whom there are a lot of outlier values
detectedOutliers.append('LOCKHART EUGENE E')
print('Detected Outliers:', detectedOutliers)

# removing outliers that were detected
for outlier in detectedOutliers:
    data_dict.pop(outlier, 0)

# Data points for which more than 85% of feature values are missing are suspected as outliers
dataKeys, data_dict_modified = suspectDataPointForWhichMoreThanXpercentFeaturesIsNaN(data_dict, 0.85)

### Create New Features
my_dataset = data_dict_modified

my_dataset, new_features = createNewFeatures(my_dataset)
all_features = all_features + new_features

#remove all NaN values
my_dataset = removeNaNvalues(my_dataset, financial_features)

# now selecting most important features using SelectKBest algo
num_features = 10
selectBestFeatures = findKbestFeatures(my_dataset, all_features, num_features)
selectedFeatures = ['poi'] + list(selectBestFeatures.keys())

print('Selected Features: {}'.format(selectedFeatures))


# Extracting features and labels from dataset corpus
data = featureFormat(my_dataset, selectedFeatures, sort_keys=True) 
labels, features = targetFeatureSplit(data)

#perform feature scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
features = scaler.fit_transform(features)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)


#### Classification
classifiers = {}

# Naive Bayes Classifier
def naive_bayes_classifier(features_train, features_test, labels_train, labels_test):
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(features_train, labels_train)
    pred = classifier.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)

    precision = precision_score(labels_test, pred)
    recall = recall_score(labels_test, pred)

    print('Naive Bayes Accuracy: {}, Precision: {}, Recall: {}'.format(accuracy, precision, recall))
    return classifier

# SVM Classifier
def svm_classifier(features_train, features_test, labels_train, labels_test):
    from sklearn.svm import SVC 
    classifier = SVC(C=1000)
    classifier.fit(features_train, labels_train)
    pred = classifier.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    precision = precision_score(labels_test, pred)
    recall = recall_score(labels_test, pred)

    print('SVM Accuracy: {}, Precision: {}, Recall: {}'.format(accuracy, precision, recall))
    return classifier


# SVM with grid search
def svm_grid_search(features_train, features_test, labels_train, labels_test):
    from sklearn.svm import SVC 
    from sklearn.model_selection import GridSearchCV
    parameters = {'kernel': ['linear', 'rbf'], 'C': [1,10,100]}
    classifier = GridSearchCV(SVC(), parameters)
    classifier.fit(features_train, labels_train)
    pred = classifier.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    precision = precision_score(labels_test, pred)
    recall = recall_score(labels_test, pred)

    print('SVM Grid Search Accuracy: {}, Precision: {}, Recall: {}'.format(accuracy, precision, recall))
    return classifier 

# Decision Tree
def decision_tree_classifier(features_train, features_test, labels_train, labels_test):
    from sklearn import tree 
    from sklearn.tree import DecisionTreeClassifier
    import graphviz 
    classifier = DecisionTreeClassifier()
    classifier.fit(features_train, labels_train)

    dot_data = tree.export_graphviz(classifier, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render('enron')

    pred = classifier.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    precision = precision_score(labels_test, pred)
    recall = recall_score(labels_test, pred)
    print('Decision Tree Accuracy: {}, Precision: {}, Recall: {}'.format(accuracy, precision, recall))

    dot_data = tree.export_graphviz(classifier, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render('poi')
    return classifier

#Adaboost Classifier
def adaboost_classifier(features_train, features_test, labels_train, labels_test):
    from sklearn.ensemble import AdaBoostClassifier
    classifier = AdaBoostClassifier(n_estimators=1000, random_state=202, learning_rate=1, algorithm='SAMME.R')
    classifier.fit(features_train, labels_train)
    pred = classifier.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    precision = precision_score(labels_test, pred)
    recall = recall_score(labels_test, pred)
    print('Adaboost Accuracy: {}, Precision: {}, Recall: {}'.format(accuracy, precision, recall))


#clf = naive_bayes_classifier(features_train, features_test, labels_train, labels_test)
#clf = svm_classifier(features_train, features_test, labels_train, labels_test)
#clf = svm_grid_search(features_train, features_test, labels_train, labels_test)
#clf = decision_tree_classifier(features_train, features_test, labels_train, labels_test)
clf = adaboost_classifier(features_train, features_test, labels_train, labels_test)



















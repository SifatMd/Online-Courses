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
    entries = data_dict.values()
    total = len(entries) 
    missingCounts = {}
    for key in entries[0].keys():
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
    data_dict_new = data_dict 

    dataKeys = []
    for name, item in data_dict.items():
        num_nan = 0
        for key, value in item.items():
            if value == "NaN":
                num_nan += 1 
        
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
            for key, val in value.items():
                if key in features and val != "NaN":
                    featureList[key].append(val)
                    nameList[key].append(name)
    
    return featureList, nameList


def findSuspectedOutliers(featureList):
    suspectedOutliers = {}
    for key in featureList.keys():
        suspectedOutliers[key] = set() 

    for key, value in featureList.items():
        q1 = np.percentile(value, 25)
        q3 = np.percentile(value, 75)
        iqr = q3 - q1 
        floor = q1 - 10*iqr 
        ceiling = q3 + 10*iqr 

        for x in value:
            if ((x<floor) | (x>ceiling)):
                suspectedOutliers[key].add(x)
    
    return suspectedOutliers


def findOutliers(data_dict, outlyingValues, threshold):
    outliers = {}
    outlyingKeys = outlyingValues.keys()

    for key in data_dict.keys():
        outliers[key] = 0 
    
    for key, value in data_dict.items():
        for k, val in value.items():
            if ((k in outlyingKeys) and (val in outlyingValues[k])):
                outliers[key] += 1 

    filteredOutliers = {}
    total = 0
    for k, v in outliers.items():
        if v>0:
            filteredOutliers[k] = v
            total += v 

    realOutliers = []
    for key, value in filteredOutliers.items():
        if value/float(total) >= threshold:
            realOutliers.append(key)
    
    return realOutliers


### Create New Features
# from poi_to_this_person_ratio and from_this_person_to_poi_ratio
def computeRatio(messages, allMessages):
    ratio = 0
    if messages == "NaN" or allMessages == "NaN":
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

    k_best = SelectKBest(f_classif, k=k)
    k_best.fit(features, labels)
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

datasetExploration(data_dict)

missingPercentage, missingValues = featuresMissingPercentage(data_dict)

displayMissingFeatures(missingPercentage, missingValues)














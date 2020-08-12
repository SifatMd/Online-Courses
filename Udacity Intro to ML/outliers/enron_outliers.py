#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "rb") )
data_dict.pop('TOTAL',0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
#print(data)

### visualizing data
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()
### outlier as it's the total column value:  [2.6704229e+07 9.7343619e+07]

### your code below
for key,val in data_dict.items():
    if val['salary']!='NaN' and val['bonus']!='NaN' and int(val['salary'])>1e6 and int(val['bonus'])>5e6:
        print('wanted person name: {}'.format(key))



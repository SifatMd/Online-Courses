#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import numpy as np 
import math 

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

print('number of persons: {}'.format(len(enron_data)))
print('number of features: {}'.format(len(enron_data['METTS MARK'])))

num_poi = 0
for key, val in enron_data.items():
    num_poi += enron_data[key]['poi']

print('number of poi: {}'.format(num_poi))

poi_file = open('../final_project/poi_names.txt', 'r')
all_lines = poi_file.readlines()
num_pois = 0
for line in all_lines:
    if line[0]=='(':
        num_pois += 1
print('total number of pois: {}'.format(num_pois))


# print('feature names {}'.format(enron_data['METTS MARK'].keys()))

print('total value of stock to James Prentice is {}'.format(enron_data['PRENTICE JAMES']['total_stock_value']))

print('Number of email from Wesley Colwell to POI is {}'.format(enron_data['COLWELL WESLEY']['from_this_person_to_poi']))

print('Exercised Stock Options by Jeffrey K Skilling is {}'.format(enron_data['SKILLING JEFFREY K']['exercised_stock_options'])) 

print('K Ley : {}, J. Skilling : {}, Andrew Fastow : {}, Max money gained: {}'.format(
    enron_data['LAY KENNETH L']['total_payments'],enron_data['SKILLING JEFFREY K']['total_payments'],enron_data['FASTOW ANDREW S']['total_payments'], 
    max(enron_data['LAY KENNETH L']['total_payments'],enron_data['SKILLING JEFFREY K']['total_payments'],enron_data['FASTOW ANDREW S']['total_payments'])
))

legit_salary = 0
known_email = 0
for key, value in enron_data.items():
    if value['salary'] != 'NaN':
        legit_salary += 1
    if value['email_address']!='NaN':
        known_email += 1

print('number of quantified salary: {}, known email address: {}'.format(legit_salary, known_email))

# # print(enron_data['SKILLING JEFFREY K'].keys())

null_payments = 0
for key, val in enron_data.items():
    if val['total_payments'] == 'NaN':
        null_payments += 1

print('Percentage of people with NaN payments: {}'.format(null_payments/len(enron_data)))






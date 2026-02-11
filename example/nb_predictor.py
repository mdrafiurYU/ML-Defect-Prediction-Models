import numpy as np
import csv as csv
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report

input_file_training = "./ant-1.9.2.csv"
input_file_test = "./ant-1.9.3.csv"

# load the training data as a matrix
dataset = pd.read_csv(input_file_training, header=0)

# separate the data from the target attributes
train_data = dataset.drop('label', axis=1)

# remove unnecessary features
train_data = train_data.drop('File', axis=1)

# the lables of training data. `label` is the title of the  last column in your CSV files
train_target = dataset.label 

# load the testing data
dataset2 = pd.read_csv(input_file_test, header=0)

# separate the data from the target attributes
test_data = dataset2.drop('label', axis=1)

# remove unnecessary features
test_data = test_data.drop('File', axis=1)

# the lables of test data
test_target = dataset2.label

#print(test_target)

gnb = GaussianNB()
#decision_t = DecisionTreeClassifier(random_state=0, max_depth=2)
test_pred = gnb.fit(train_data, train_target).predict(test_data)

print(classification_report(test_target, test_pred, labels=[0,1]))

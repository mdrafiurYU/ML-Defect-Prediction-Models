"""
dtc.py — Decision Tree Classifier with Max-Depth Hyperparameter Sweep

This script is a variant of decision_tree_classifier.py that systematically
evaluates different max_depth values for the Decision Tree classifier. For
each max_depth setting, it performs 6-fold cross-validation and reports the
aggregated F1-score. This helps identify the optimal tree depth that balances
model complexity against predictive performance.

Usage:
    python dtc.py --name <project_name>

Arguments:
    --name  Name of the project directory containing the data splits
            (e.g., 'jdt', 'lucene', 'xorg', 'jackrabbit').

Output:
    Prints the F1-score for each max_depth setting tested.
"""

import argparse
import numpy as np
import csv as csv
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import decomposition, datasets
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler

# ---- Command-line argument parsing ----
parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, type=str, help="Project name")
args = parser.parse_args()
project_name = args.name

# List of max_depth values to evaluate
max_depths = [2,4,6,8,12,20,30,50,80,100]

# ---- Outer loop: iterate over each max_depth hyperparameter ----
for max_depth in max_depths:

    # Reset confusion matrix accumulators for each depth setting
    count = 0
    tp_total = 0
    fp_total = 0
    fn_total = 0

    # ---- Inner loop: 6-fold cross-validation ----
    while(count < 6):
        dir_num = str(count)
        input_file_training = project_name + "/" + dir_num + "/train.csv"
        input_file_test = project_name + "/" + dir_num + "/test.csv"

        # Load training data from CSV into a pandas DataFrame
        dataset = pd.read_csv(input_file_training, header=0)

        # Separate features from the target label column ('500_Buggy?')
        train_data = dataset.drop('500_Buggy?', axis=1)

        # Drop non-predictive metadata columns
        train_data = train_data.drop(['change_id', '411_commit_time', '412_full_path'], axis=1)

        # Extract the target labels (last column: '500_Buggy?')
        train_target = dataset[dataset.columns[-1]]

        # Load test data from CSV
        dataset2 = pd.read_csv(input_file_test, header=0)

        # Separate features from the target label for test data
        test_data = dataset2.drop('500_Buggy?', axis=1)

        # Drop the same non-predictive columns from the test data
        test_data = test_data.drop(['change_id', '411_commit_time', '412_full_path'], axis=1)

        # Extract the target labels for test data
        test_target = dataset2[dataset2.columns[-1]]

        # Train a Decision Tree with entropy criterion and the current max_depth
        decision_t = DecisionTreeClassifier(criterion = 'entropy', max_depth = max_depth)
        test_pred = decision_t.fit(train_data, train_target).predict(test_data)

        # Extract confusion matrix components for this fold
        tn, fp, fn, tp = confusion_matrix(test_target, test_pred).ravel()
    
        # Accumulate confusion matrix values across folds
        tp_total = tp_total + tp
        fp_total = fp_total + fp
        fn_total = fn_total + fn
        count = count + 1

    # ---- Compute and print the F1-score for this max_depth setting ----
    precision = tp_total / (tp_total + fp_total)
    recall = tp_total / (tp_total + fn_total)
    f1 = (2 * precision * recall) / (precision + recall)
    print("f1-score for project " + project_name + " with max_depth: " + str(max_depth) + " is: " + str(f1))

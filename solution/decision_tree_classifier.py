"""
decision_tree_classifier.py — Defect Prediction using Decision Tree Classifier

This script trains a Decision Tree classifier (using entropy as the split
criterion) to predict whether a software change is buggy or not. It performs
6-fold cross-validation by iterating through numbered subdirectories (0–5),
each containing a train.csv and test.csv split. The script aggregates
true positives, false positives, and false negatives across all folds to
compute an overall F1-score.

Usage:
    python decision_tree_classifier.py --name <project_name>

Arguments:
    --name  Name of the project directory containing the data splits
            (e.g., 'jdt', 'lucene', 'xorg', 'jackrabbit').

Output:
    Prints the aggregated F1-score across all 6 folds.
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

# ---- Initialize accumulators for confusion matrix values across all folds ----
count = 0        # Current fold index (0 through 5)
tp_total = 0     # Accumulated true positives
fp_total = 0     # Accumulated false positives
fn_total = 0     # Accumulated false negatives

# ---- 6-Fold Cross-Validation Loop ----
while(count < 6):
    dir_num = str(count)
    input_file_training = project_name + "/" + dir_num + "/train.csv"
    input_file_test = project_name + "/" + dir_num + "/test.csv"

    # Load training data from CSV into a pandas DataFrame
    dataset = pd.read_csv(input_file_training, header=0)

    # Separate features from the target label column ('500_Buggy?')
    train_data = dataset.drop('500_Buggy?', axis=1)

    # Drop non-predictive metadata columns (change ID, commit time, file path)
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

    #######################
    # Hyperparameter candidates (defined but not used in this version;
    # the classifier runs with default max_depth instead of grid search)
    best_accuracy = 0.0
    best_depth = 0
    max_depths = [2,4,6,8,12,20,30,50,80,100]
    ########################
  
    # Train a Decision Tree classifier using entropy criterion and predict on test set
    decision_t = DecisionTreeClassifier(criterion = 'entropy')
    test_pred = decision_t.fit(train_data, train_target).predict(test_data)

    # Extract confusion matrix components for this fold
    tn, fp, fn, tp = confusion_matrix(test_target, test_pred).ravel()
    
    # Accumulate confusion matrix values across folds
    tp_total = tp_total + tp
    fp_total = fp_total + fp
    fn_total = fn_total + fn
    count = count + 1

# ---- Compute and print the overall F1-score across all 6 folds ----
precision = tp_total / (tp_total + fp_total)
recall = tp_total / (tp_total + fn_total)
f1 = (2 * precision * recall) / (precision + recall)
print("f1-score for project " + project_name + " is: " + str(f1))

"""
random_forest.py — Defect Prediction using Random Forest with Grid Search
                   Hyperparameter Tuning

This script trains a Random Forest classifier with exhaustive hyperparameter
optimization via GridSearchCV. It searches over n_estimators, max_depth,
min_samples_split, and min_samples_leaf to find the best-performing model
for software defect prediction. The script iterates through 6 pre-split
data folds and aggregates the results to compute an overall F1-score.

Usage:
    python random_forest.py --name <project_name>

Arguments:
    --name  Name of the project directory containing the data splits
            (e.g., 'jdt', 'lucene', 'xorg', 'jackrabbit').

Output:
    Prints per-fold accuracy and confusion matrix, followed by the
    overall aggregated F1-score.
"""

import argparse
import numpy as np
import csv as csv
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn import metrics

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

    # (Unused) Alternative: a manually configured Random Forest model
    #model = RandomForestClassifier(n_estimators=700, bootstrap = True, max_features = 'sqrt', random_state = 1)

    # (Unused) Initial param_grid — replaced by the more comprehensive 'hyperF' grid below
    param_grid = {
        'bootstrap': [True],
        'max_depth': [2, 4, 8, 12, 20, 30, 50, 80, 100],
        'n_estimators': [10,30,50,120,150,200,250]
    }

    # Create the base Random Forest model for tuning
    rf = RandomForestClassifier()

    # (Unused) Alternative model with entropy criterion
    forest = RandomForestClassifier(criterion = "entropy")

    # Define the hyperparameter search space for GridSearchCV
    n_estimators = [10, 30,50,120,150,200]
    max_depth = [4, 5, 8, 15, 25, 30, 50,80,100]
    min_samples_split = [2, 5, 10, 15, 100]
    min_samples_leaf = [1, 2, 5, 10] 

    hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)

    # Run GridSearchCV with 4-fold internal CV and parallel execution
    grid = GridSearchCV(rf, hyperF, cv = 4, verbose = 1, n_jobs = -1)
    test_pred = grid.fit(train_data, train_target).predict(test_data)

    # Print per-fold accuracy and confusion matrix breakdown
    print(metrics.accuracy_score(test_target, test_pred))
    tn, fp, fn, tp = confusion_matrix(test_target, test_pred).ravel()
    print(tn, fp, fn, tp)

    # Accumulate confusion matrix values across folds
    tp_total = tp_total + tp
    fp_total = fp_total + fp
    fn_total = fn_total + fn
    count = count + 1

# ---- Compute and print the overall F1-score across all 6 folds ----
precision = tp_total / (tp_total + fp_total)
recall = tp_total / (tp_total + fn_total)
f1 = (2 * precision * recall) / (precision + recall)
print(f1)

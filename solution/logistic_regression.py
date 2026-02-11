"""
logistic_regression.py — Defect Prediction using Logistic Regression with
                         Grid Search Hyperparameter Tuning

This script trains a Logistic Regression classifier with automated
hyperparameter optimization via GridSearchCV. It searches over L2
regularization strength (C), solver type, and max iterations to find
the best model for predicting buggy software changes. The script
performs 6-fold cross-validation across pre-split data directories
and reports the aggregated F1-score.

Usage:
    python logistic_regression.py --name <project_name>

Arguments:
    --name  Name of the project directory containing the data splits
            (e.g., 'jdt', 'lucene', 'xorg', 'jackrabbit').

Output:
    Prints per-fold confusion matrix components (tn, fp, fn, tp)
    and the overall aggregated F1-score.
"""

import argparse
import numpy as np
import csv as csv
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model, decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

    # Initialize the Logistic Regression model
    logistic = LogisticRegression()

    # Define the hyperparameter grid for GridSearchCV:
    #   - penalty: L2 regularization
    #   - C: inverse regularization strength (log-spaced from 1e-4 to 1e4)
    #   - solver: liblinear (good for small datasets)
    #   - max_iter: number of iterations for convergence
    param_grid = [    
    {'penalty' : ['l2'],
        'C' : np.logspace(-4, 4, 20),
        'solver' : ['liblinear'],
        'max_iter' : [10, 20, 30, 50, 80, 100]
        }
    ]

    # Run GridSearchCV with 4-fold internal CV and parallel execution
    clf = GridSearchCV(logistic, param_grid = param_grid, cv = 4, verbose=1, n_jobs=-1)
    test_pred = clf.fit(train_data, train_target).predict(test_data)

    # Extract confusion matrix components for this fold
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

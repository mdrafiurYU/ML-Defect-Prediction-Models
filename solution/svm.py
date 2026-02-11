"""
svm.py — Defect Prediction using Support Vector Machine (SVM) with
         Grid Search Hyperparameter Tuning

This script trains a Support Vector Machine classifier with automated
hyperparameter optimization via GridSearchCV. It searches over kernel
type (polynomial and RBF), degree (for poly), C (regularization), and
gamma (kernel coefficient) to find the best model for predicting buggy
software changes. The script performs 6-fold cross-validation across
pre-split data directories and reports the aggregated F1-score.

Usage:
    python svm.py --name <project_name>

Arguments:
    --name  Name of the project directory containing the data splits
            (e.g., 'jdt', 'lucene', 'xorg', 'jackrabbit').

Output:
    Prints the project name, per-fold confusion matrix components,
    and the overall aggregated F1-score.
"""

import numpy as np
import csv as csv
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import argparse
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit

# ---- Command-line argument parsing ----
parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, type=str, help="Project name")
args = parser.parse_args()
project_name = args.name

print(project_name)

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
    
    # Define the initial hyperparameter grid for GridSearchCV:
    #   - Polynomial kernel: sweep degree from 0 to 6
    #   - RBF kernel: sweep C and gamma across orders of magnitude
    param_grid = [
        {'degree': [0, 1, 2, 3, 4, 5, 6], 'kernel': ['poly']},
        {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ['rbf']},
    ]

    # Initialize GridSearchCV with SVC, using the parameter grid above
    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 2)
    
    # (Overwrites param_grid above — only RBF kernel with log-spaced C and gamma)
    kern = ['rbf']
    C_range = np.logspace(-2, 4, 13)
    gamma_range = np.logspace(-3, 3, 13)
    param_grid = dict(kernel=kern, gamma=gamma_range, C=C_range)

    # Fit the grid search on training data and predict on the test set
    test_pred = grid.fit(train_data, train_target).predict(test_data)

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

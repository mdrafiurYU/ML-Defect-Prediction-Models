# Software Defect Prediction Using Modified Source Code

A machine learning pipeline for predicting software defects by analyzing source code changes (patches) from open-source Java projects. The system extracts Bag-of-Words features from deleted lines in patch files and trains multiple classifiers to predict whether a given code change introduces a bug.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Feature Extraction](#feature-extraction)
- [Machine Learning Models](#machine-learning-models)
- [Usage](#usage)
- [Requirements](#requirements)

---

## Overview

This project implements a **change-level defect prediction** approach. Instead of predicting whether an entire file is buggy, it predicts whether a specific code change (commit/patch) is likely to introduce a defect. The workflow is:

1. **Feature Extraction** — Parse Java patch files to build Bag-of-Words (BoW) feature vectors from deleted source code lines.
2. **Model Training & Evaluation** — Train multiple ML classifiers using 6-fold cross-validation and evaluate using Precision, Recall, and F1-score.

---

## Project Structure

```
solution/
├── bow.py                        # Bag-of-Words feature extractor for patch files
├── decision_tree_classifier.py   # Decision Tree classifier (default depth)
├── dtc.py                        # Decision Tree with max_depth sweep
├── logistic_regression.py        # Logistic Regression with GridSearchCV
├── nb_predictor.py               # Gaussian Naive Bayes classifier (baseline)
├── random_forest.py              # Random Forest with GridSearchCV
├── rfc.py                        # Random Forest with max_depth sweep
├── svm.py                        # Support Vector Machine with GridSearchCV
├── train.csv                     # Sample training data
├── test.csv                      # Sample test data
├── jdt/                          # Eclipse JDT project data
│   ├── 0/ ... 5/                 # 6-fold cross-validation splits (train.csv, test.csv)
│   ├── patch/                    # Raw patch files (.patch) for feature extraction
│   └── patch.zip                 # Compressed archive of patch files
├── jackrabbit/                   # Apache Jackrabbit project data
├── lucene/                       # Apache Lucene project data
└── xorg/                         # X.Org project data
```

---

## Dataset

The project uses data from four well-known open-source Java projects:

| Project        | Description                                      |
|----------------|--------------------------------------------------|
| **JDT**        | Eclipse Java Development Tools                   |
| **Jackrabbit** | Apache Jackrabbit content repository              |
| **Lucene**     | Apache Lucene search engine library               |
| **Xorg**       | X.Org display server                             |

Each project directory contains:
- **Fold directories (`0/` – `5/`)** — Pre-split train/test CSV files for 6-fold cross-validation.
- **`patch/` directory** — Raw `.patch` files representing individual code changes, used by `bow.py` for feature extraction.

### CSV Data Format

Each CSV file contains columns including:
- `change_id` — Unique identifier for the code change
- `411_commit_time` — Timestamp of the commit
- `412_full_path` — File path of the changed file
- Various numeric feature columns (code metrics)
- `500_Buggy?` — Target label (`1` = buggy, `0` = clean)

---

## Feature Extraction

### `bow.py` — Bag-of-Words Extractor

Extracts textual features from Java patch files:

1. Reads all `.patch` files from a project's `patch/` directory.
2. Isolates **deleted lines** (lines starting with `-`) which represent the original buggy code.
3. Filters out diff headers, comments, and Java keywords.
4. Tokenizes the remaining code into individual identifiers.
5. Builds a global vocabulary using `CountVectorizer`.
6. Generates a per-patch BoW feature vector and writes results to `bow.csv`.

> **Note:** The patch directory path is currently hardcoded to `./jdt/patch/*.patch`. Modify the `path` variable to process other projects.

---

## Machine Learning Models

All classifiers use the same evaluation strategy:
- **6-fold cross-validation** across pre-split data directories (`0/` – `5/`)
- Aggregated **Precision**, **Recall**, and **F1-score** computed from cumulative confusion matrix values

### Models

| Script                          | Algorithm              | Hyperparameter Strategy                          |
|---------------------------------|------------------------|--------------------------------------------------|
| `nb_predictor.py`               | Gaussian Naive Bayes   | No tuning (baseline)                             |
| `decision_tree_classifier.py`   | Decision Tree          | Entropy criterion, default depth                 |
| `dtc.py`                        | Decision Tree          | Max-depth sweep: `[2,4,6,8,12,20,30,50,80,100]` |
| `logistic_regression.py`        | Logistic Regression    | GridSearchCV (L2 penalty, C, solver, max_iter)   |
| `random_forest.py`              | Random Forest          | GridSearchCV (n_estimators, max_depth, etc.)     |
| `rfc.py`                        | Random Forest          | Max-depth sweep with fixed `n_estimators=30`     |
| `svm.py`                        | Support Vector Machine | GridSearchCV (poly + RBF kernels, C, gamma)      |

---

## Usage

### 1. Feature Extraction

Generate BoW features from patch files (edit the `path` variable inside `bow.py` for the desired project):

```bash
cd solution
python bow.py
```

This produces a `bow.csv` file with BoW feature vectors.

### 2. Train & Evaluate Models

Each classifier script accepts a `--name` argument specifying the project directory:

```bash
# Gaussian Naive Bayes
python nb_predictor.py --name jdt

# Decision Tree (default depth)
python decision_tree_classifier.py --name jdt

# Decision Tree (max_depth sweep)
python dtc.py --name jdt

# Logistic Regression with Grid Search
python logistic_regression.py --name jdt

# Random Forest with Grid Search
python random_forest.py --name jdt

# Random Forest (max_depth sweep)
python rfc.py --name jdt

# Support Vector Machine with Grid Search
python svm.py --name jdt
```

Replace `jdt` with `jackrabbit`, `lucene`, or `xorg` to evaluate on other projects.

---

## Requirements

- **Python** 3.6+
- **Dependencies:**

```
numpy
pandas
scikit-learn
matplotlib
```

Install all dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## License

This project is provided for academic and research purposes.

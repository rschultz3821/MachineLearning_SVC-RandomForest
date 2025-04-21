# SVC & Random Forests
## Overview
This repository contains three supervised machine learning projects exploring classification using:

* Decision Trees on a car purchasing dataset
* Ensemble methods (Random Forest, Bagging with SVC) on an online gaming behavior dataset
* SVC and Random Forest on a crab age prediction dataset

Each project includes feature engineering, model training, hyperparameter tuning using GridSearchCV, model evaluation, and visualization.

### 1. Car Purchase Classification
#### Dataset: Sales Prediction Dataset on Kaggle
#### Goal: Predict whether a customer falls into a "Low", "Medium", or "High" car purchase category.

#### Key Steps:
Features: gender, annual salary, credit card debt, net worth, age

Target: purchase_category (Low/Medium/High) from car purchase amount

Model: DecisionTreeClassifier

Hyperparameter Tuning: Best max_depth using GridSearchCV

Visualization: Decision Tree plotted using plot_tree

### 2. Online Gaming Behavior Prediction
#### Dataset: Online Gaming Behavior Dataset on Kaggle
#### Goal: Predict user engagement level based on gameplay metrics.

#### Models Used:
SVC with RBF kernel, optimized with GridSearchCV for C and gamma

Random Forest, optimized for best max_depth

Bagging Classifier using LinearSVC as base estimator

#### Evaluation:
Accuracy scores on test set

Confusion matrices for all models

OOB scores for ensemble methods

Feature importance visualization for Random Forest

### 3. Crab Sex Prediction
#### Dataset: Crab Age Prediction on Kaggle
#### Goal: Predict crab sex based on physical characteristics.

#### Models Used:
SVC with RBF kernel: tuned using C and gamma

Random Forest: best max_depth determined via GridSearchCV

#### Evaluation:
Train/test accuracy and OOB scores

Confusion matrix for Random Forest

Comparison of SVC vs Random Forest performance

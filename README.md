# SVC & Random Forests
## Overview
This repository contains three supervised machine learning projects exploring classification using:

* Decision Trees on a car purchasing dataset
* Random Forest and Bagging with SVC on an online gaming behavior dataset
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

#### Results: 
![image](https://github.com/user-attachments/assets/9d3b9871-d0a2-47ba-9d03-7e2bf745b928)
![image](https://github.com/user-attachments/assets/800796ce-98c9-45b0-a933-254743e4f473)

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

#### Results
![image](https://github.com/user-attachments/assets/1076fdfd-3c90-46df-a3fb-b23ab463057a)
![image](https://github.com/user-attachments/assets/c0f89358-6713-402a-a1a1-e30cf191524a)
![image](https://github.com/user-attachments/assets/487c2422-c073-4c11-a1a5-0308af918d7e)
![image](https://github.com/user-attachments/assets/70d1e163-e6f4-46b2-8913-00e0e2073d6a)
![image](https://github.com/user-attachments/assets/615e9d4c-e3c5-4bb7-bcc4-d956fa70922e)
![image](https://github.com/user-attachments/assets/def46838-94c5-4f27-808f-897fd47739c1)

Both models perform very similarly on the test set, with Random Forest slightly outperforming SVC (Bagging) by ~0.1%. These are high accuracy scores, indicating that both models are well-suited to the task. OOB accuracy of 1.000 for both models is unusually high and likely indicates:

*Overfitting: The models may have memorized the training data.

*Small dataset: If the dataset is very small, OOB samples might not be representative.

*Data leakage: Information from the test set or labels may have inadvertently influenced training.

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

#### Results
![image](https://github.com/user-attachments/assets/083d1744-2aec-4cfe-ba5d-190d264808e1)
![image](https://github.com/user-attachments/assets/3007799b-6a6d-438d-b3bf-147c7b6e77fe)
The SVC is performing slightly better than Random Forest.


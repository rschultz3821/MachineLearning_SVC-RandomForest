import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC

#region Read in dataset
#https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset
df = pd.read_csv("online_gaming_behavior_dataset.csv").sample(3000)
df = df.dropna()

df['GameDifficulty'] = df['GameDifficulty'].replace({'Easy': 0, 'Medium': 1, 'Hard': 2})
df['GameDifficulty'] = df['GameDifficulty'].astype(float)
df.info()

# Feature columns
X = df.iloc[:, 5:-1]
X = (X - np.average(X, axis=0)) / np.std(X, axis=0)
# Target variable - engagement level
y = df.iloc[:, -1]
#endregion

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#region Use a grid search to find an optimal SVC
clf = SVC(kernel="rbf")
clf.fit(X_train, y_train)
parameters = {"C": np.linspace(10, 100, num=10),
"gamma": np.linspace(0.01, .1, num=10)}

grid_search = GridSearchCV(clf, param_grid=parameters, cv=5)
grid_search.fit(X_train, y_train)
C_best = grid_search.best_params_["C"]
gamma_best = grid_search.best_params_["gamma"]

clf = SVC(kernel="rbf", C=C_best, gamma=gamma_best)

clf.fit(X_train, y_train)

print("-"*20, " SVC Results ", "-"*20)
print(f"Best C: ", C_best)
print(f"Best Gamma: ", gamma_best)

score_df = pd.DataFrame(grid_search.cv_results_)
print(score_df[['param_C', 'param_gamma', 'mean_test_score', 'rank_test_score']])

test_score_SVC = clf.score(X_test, y_test)
#endregion

#region Use a grid search to find optimal random forest max_depth
clf = RandomForestClassifier(oob_score=True, verbose=3)
clf.fit(X_train, y_train)
parameters = {"max_depth": range(2,20)}
grid_search = GridSearchCV(clf, param_grid=parameters, cv=5)
grid_search.fit(X_train, y_train)
max_depth_best = grid_search.best_params_["max_depth"]

clf = RandomForestClassifier(oob_score=True, max_depth=max_depth_best, verbose=3)

clf.fit(X_train, y_train)

print("-"*20, " Random Forest Results ", "-"*20)
print(f"Best max_depth: ", max_depth_best)

score_df = pd.DataFrame(grid_search.cv_results_)
print(score_df[['param_max_depth', 'mean_test_score', 'rank_test_score']])

test_score_RF = clf.score(X_test, y_test)
#endregion

#region Use a grid search to find optimal bagging classifier n_estimators
print("-" * 20, " Training BaggingClassifier with LinearSVC ", "-" * 20)
base_svc = LinearSVC(max_iter=10000)  # Linear SVM for Bagging
bagging_clf = BaggingClassifier(estimator=base_svc, n_jobs=-1, oob_score=True)

parameters = {"n_estimators": [10, 50, 100]}

grid_search = GridSearchCV(bagging_clf, param_grid=parameters, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters
best_n_estimators = grid_search.best_params_["n_estimators"]

# Train final BaggingClassifier model with best parameters
bagging_clf = BaggingClassifier(estimator=base_svc, n_estimators=best_n_estimators, n_jobs=-1, oob_score=True)
bagging_clf.fit(X_train, y_train)

test_score_bagging = bagging_clf.score(X_test, y_test)

print("-" * 20, " BaggingClassifier Results ", "-" * 20)
print(f"Best n_estimators: {best_n_estimators}")
print(f"BaggingClassifier Test Score: {test_score_bagging:.4f}")
#endregion

#region Confusion Matrix for BaggingClassifier
cm_bagging = confusion_matrix(y_test, bagging_clf.predict(X_test), normalize="true")
disp_cm_bagging = ConfusionMatrixDisplay(cm_bagging, display_labels=np.unique(y))
disp_cm_bagging.plot()
plt.title("Confusion Matrix - Bagging Classifier")
plt.show()
print(f"OOB Score: {bagging_clf.oob_score:.3f}")
#endregion

#region Matrix for Random Forest
cm_svc = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
disp_cm_svc = ConfusionMatrixDisplay(cm_svc, display_labels=np.unique(y))
disp_cm_svc.plot()
plt.title("Confusion Matrix - Random Forest")
plt.show()
print(f"OOB Score: {clf.oob_score_:.3f}")
#endregion

#Running against whole dataset
clf.fit(X, y)  # Random Forest
bagging_clf.fit(X, y)  # Bagging Classifier

#region Scores
print("-"*20, " Comparing Scores ", "-"*20)
print("SVC (Bagging) Test Score: ", test_score_SVC)
print("Random Forest Test Score: ", test_score_RF)

print(f"OOB Score SVC (Bagging): {bagging_clf.oob_score:.3f}")
print(f"OOB Score Random Forest: {clf.oob_score:.3f}")

#endregion

#region Confusion Matrix for BaggingClassifier
cm_bagging = confusion_matrix(y_test, bagging_clf.predict(X_test), normalize="true")
disp_cm_bagging = ConfusionMatrixDisplay(cm_bagging, display_labels=np.unique(y))
disp_cm_bagging.plot()
plt.title("Bagging Classifier Training on Entire Dataset")
plt.show()
#endregion

#region Importance features and Matrix for Random Forest
importances = pd.DataFrame(clf.feature_importances_, index=df.columns[5:-1])
importances.plot.bar(figsize=(15, 6))
plt.xticks(rotation=10)
plt.show()

cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true") #doesn't work for precision or recall so we can compare
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.title("Random Forest Training on Entire Dataset")
plt.show()
#endregion

print("Random Forest is more interpretable due to feature importances, and performs well with less tuning.")
print("Bagging with LinearSVC is effective at improving stability, but it doesn't provide feature importance for interpretation.")


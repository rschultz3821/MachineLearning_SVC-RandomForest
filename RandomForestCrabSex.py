import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#region Load dataset
# https://www.kaggle.com/datasets/sidhus/crab-age-prediction
df = pd.read_csv("CrabAgePrediction.csv")

df = df[df['Sex'] != 'M']

X = df.iloc[:, 1:].copy().to_numpy()
X = (X - np.average(X, axis=0)) / np.std(X, axis=0)
y = df.iloc[:, 0].copy().to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) #Training on 70%
#endregion

# region Grid search to find an optimal SVC
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

# region Optimal Random Forest
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
#endregion

#region Compare results
test_score_RF = clf.score(X_test, y_test)

print("-"*20, " Comparing Scores ", "-"*20)
print("SVC Test Score: ", test_score_SVC)
print("Random Forest Test Score: ", test_score_RF)
print(f"Random Forest Train Score: {clf.score(X_train, y_train):.4f}")
print(f"Random Forest Test Score: {clf.score(X_test, y_test):.4f}")
print(f"OOB Score: {clf.oob_score_:.3f}")
#endregion

# region Confusion matrix for the Random Forest Classifier
cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.title("Confusion Matrix - Random Forest")
plt.show()
#endregion

print("SVC (RBF) \n"
      "Good for complex, non-linear boundaries. Performs well with fewer outliers.	\n"
      "Can be slow on large datasets. Sensitive to hyperparameters like C and gamma.\n\n"
      "Random Forest	\n"
      "Handles large datasets well. Less prone to overfitting. \n"
      "Can provide feature importance.	May not capture complex boundaries as precisely as SVC.")

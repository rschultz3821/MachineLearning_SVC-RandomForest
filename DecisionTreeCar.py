import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

#region Load Dataset
# https://www.kaggle.com/datasets/mohdshahnawazaadil/sales-prediction-dataset
df = pd.read_csv("car_purchasing.csv", encoding='latin-1')

# Create a categorical label for car purchase amount
df['purchase_category'] = pd.qcut(df['car purchase amount'], q=3, labels=['Low', 'Medium', 'High'])

# Features
X = df[["gender", "annual Salary", "credit card debt", "net worth", "age"]].copy()

# Target variable - categorical
y = df["purchase_category"]
#endregion

# region Train Initial Tree with max_depth 3
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)
print(f"Training Accuracy (depth 3): {clf.score(X, y):.3f}")

plt.figure(figsize=(15, 8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=clf.classes_)
plt.title("Decision Tree (Max Depth = 3)")
plt.show()
#endregion

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# region Grid Search for Optimal Depth
param_grid = {"max_depth": range(2, 16)}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_max_depth = grid_search.best_params_["max_depth"]
print(f"Best Max Depth: {best_max_depth}")
#endregion

# Retrain with best depth
clf_best = DecisionTreeClassifier(max_depth=best_max_depth, random_state=42)
clf_best.fit(X_train, y_train)

# Print Scores
print(f"Training Accuracy: {clf_best.score(X_train, y_train):.3f}")
print(f"Test Accuracy: {clf_best.score(X_test, y_test):.3f}")

# Print Optimal max_depth
score_df = pd.DataFrame(grid_search.cv_results_)
print(score_df[['param_max_depth', 'mean_test_score', 'rank_test_score']])

print("The target classes (Low, Medium, High purchase categories) are already \n"
      "relatively balanced after applying pd.qcut() the variable into three equal categories.")


# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    ConfusionMatrixDisplay
)

# Load dataset
data = pd.read_csv('creditcard.csv')  # Make sure this matches your file name

# Overview
print(data.head())
print(data.info())
print(data.describe())

# Data types
obj = (data.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:", len(object_cols))

int_ = (data.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:", len(num_cols))

fl = (data.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:", len(fl_cols))

# Check fraud distribution
print(data['Class'].value_counts())

# Distribution plot
plt.figure(figsize=(12, 6))
sns.histplot(data['Amount'], bins=50, kde=True)
plt.title("Transaction Amount Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(data.corr(), cmap='BrBG', linewidths=2, annot=False)
plt.title("Correlation Heatmap")
plt.show()

# Feature and target
X = data.drop(['Class', 'Time'], axis=1)
y = data['Class']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Decision Tree Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Probabilities for ROC AUC
train_proba = model.predict_proba(X_train)[:, 1]
test_proba = model.predict_proba(X_test)[:, 1]

# Evaluation
print("Training Accuracy Score :", accuracy_score(y_train, train_preds))
print("Validation Accuracy Score :", accuracy_score(y_test, test_preds))

print("Training ROC AUC Score :", roc_auc_score(y_train, train_proba))
print("Validation ROC AUC Score :", roc_auc_score(y_test, test_proba))

# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
plt.title("Confusion Matrix - Decision Tree")
plt.show()

# Feature Importances
importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=importances[:10], y=importances.index[:10])
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()
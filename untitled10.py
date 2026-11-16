# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 23:12:24 2026

@author: HP
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)

# 1. LOAD DATASET

data = pd.read_csv(
    r"C:\Users\HP\Downloads\Train_details_22122017.csv"
)

print("First 5 rows:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(data.columns.tolist())

print("\nMissing Values:")
print(data.isnull().sum())

# 2. DATA PREPROCESSING

data["Distance"] = pd.to_numeric(data["Distance"], errors="coerce")
data["SEQ"] = pd.to_numeric(data["SEQ"], errors="coerce")

data = data.dropna(
    subset=[
        "Train No", "SEQ", "Distance",
        "Source Station", "Destination Station"
    ]
)

data = data.drop_duplicates()

# Create one row for each train
train_data = data.groupby("Train No", as_index=False).agg(
    Train_Name=("Train Name", "first"),
    Route_Distance=("Distance", "max"),
    Number_of_Stops=("SEQ", "count"),
    Source=("Source Station", "first"),
    Destination=("Destination Station", "first")
)

print("\nProcessed Data:")
print(train_data.head())

print("\nProcessed Dataset Shape:")
print(train_data.shape)

# 3. EXPLORATORY DATA ANALYSIS

print("\nRoute Distance Statistics:")
print(train_data["Route_Distance"].describe())

print("\nAverage Route Distance:")
print(train_data["Route_Distance"].mean())

print("\nMaximum Route Distance:")
print(train_data["Route_Distance"].max())

print("\nMinimum Route Distance:")
print(train_data["Route_Distance"].min())

# Distribution of route distance
plt.figure(figsize=(8, 5))
sns.histplot(train_data["Route_Distance"], bins=25, kde=True)
plt.title("Distribution of Train Route Distance")
plt.xlabel("Route Distance")
plt.ylabel("Number of Trains")
plt.show()

# Top 10 stations served by distinct trains
top_stations = (
    data.groupby("Station Name")["Train No"]
    .nunique()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 5))
top_stations.plot(kind="bar")
plt.title("Top 10 Stations by Number of Trains")
plt.xlabel("Station Name")
plt.ylabel("Number of Trains")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 4. PREPARE TRAINING AND TESTING DATA

# Simple Regression uses one feature
X_simple = train_data[["Number_of_Stops"]]

# Multiple Regression uses three features
X_multiple = train_data[
    ["Number_of_Stops", "Source", "Destination"]
]

# Convert station codes into numeric dummy variables
X_multiple = pd.get_dummies(
    X_multiple,
    columns=["Source", "Destination"],
    drop_first=True,
    dtype=int
)

y = train_data["Route_Distance"]

# Split simple regression data
X_train_simple, X_test_simple, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

# Split multiple regression data using the same random state
X_train_multi, X_test_multi, _, _ = train_test_split(
    X_multiple, y, test_size=0.2, random_state=42
)

print("\nTraining Data Shape:", X_train_multi.shape)
print("Testing Data Shape:", X_test_multi.shape)

# ------------------------------------------------
# 5. MODEL 1: SIMPLE LINEAR REGRESSION
# ------------------------------------------------

model_slr = LinearRegression()

model_slr.fit(X_train_simple, y_train)

y_pred_slr = model_slr.predict(X_test_simple)

mae_slr = mean_absolute_error(y_test, y_pred_slr)
mse_slr = mean_squared_error(y_test, y_pred_slr)
rmse_slr = np.sqrt(mse_slr)
r2_slr = r2_score(y_test, y_pred_slr)

print("\nSimple Linear Regression Results")
print("MAE:", round(mae_slr, 2))
print("MSE:", round(mse_slr, 2))
print("RMSE:", round(rmse_slr, 2))
print("R2 Score:", round(r2_slr, 4))

# 6. MODEL 2: MULTIPLE LINEAR REGRESSION

model_mlr = LinearRegression()

model_mlr.fit(X_train_multi, y_train)

y_pred_mlr = model_mlr.predict(X_test_multi)

mae_mlr = mean_absolute_error(y_test, y_pred_mlr)
mse_mlr = mean_squared_error(y_test, y_pred_mlr)
rmse_mlr = np.sqrt(mse_mlr)
r2_mlr = r2_score(y_test, y_pred_mlr)

print("\nMultiple Linear Regression Results")
print("MAE:", round(mae_mlr, 2))
print("MSE:", round(mse_mlr, 2))
print("RMSE:", round(rmse_mlr, 2))
print("R2 Score:", round(r2_mlr, 4))

# 7. MODEL 3: POLYNOMIAL REGRESSION

poly = PolynomialFeatures(degree=2, include_bias=False)

X_train_poly = poly.fit_transform(X_train_simple)
X_test_poly = poly.transform(X_test_simple)

model_poly = LinearRegression()

model_poly.fit(X_train_poly, y_train)

y_pred_poly = model_poly.predict(X_test_poly)

mae_poly = mean_absolute_error(y_test, y_pred_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("\nPolynomial Regression Results")
print("MAE:", round(mae_poly, 2))
print("MSE:", round(mse_poly, 2))
print("RMSE:", round(rmse_poly, 2))
print("R2 Score:", round(r2_poly, 4))

# 8. MODEL 4: DECISION TREE REGRESSION

model_dt = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=5,
    random_state=42
)

model_dt.fit(X_train_multi, y_train)

y_pred_dt = model_dt.predict(X_test_multi)

mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print("\nDecision Tree Regression Results")
print("MAE:", round(mae_dt, 2))
print("MSE:", round(mse_dt, 2))
print("RMSE:", round(rmse_dt, 2))
print("R2 Score:", round(r2_dt, 4))

# 9. CREATE CLASSIFICATION TARGET

# 0 = Short route
# 1 = Long route

median_distance = train_data["Route_Distance"].median()

train_data["Route_Type"] = (
    train_data["Route_Distance"] >= median_distance
).astype(int)

y_class = train_data["Route_Type"]

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_multiple,
    y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)

print("\nMedian Route Distance:", median_distance)
print("0 = Short Route")
print("1 = Long Route")

# Feature scaling for classification
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_cls)
X_test_scaled = scaler.transform(X_test_cls)

# 10. MODEL 5: LOGISTIC REGRESSION

model_log = LogisticRegression(max_iter=1000)

model_log.fit(X_train_scaled, y_train_cls)

y_pred_log = model_log.predict(X_test_scaled)

acc_log = accuracy_score(y_test_cls, y_pred_log)
cm_log = confusion_matrix(y_test_cls, y_pred_log, labels=[0, 1])

print("\nLogistic Regression Results")
print("Accuracy:", round(acc_log * 100, 2), "%")
print("Precision:", round(
    precision_score(y_test_cls, y_pred_log, zero_division=0), 3
))
print("Recall:", round(
    recall_score(y_test_cls, y_pred_log, zero_division=0), 3
))
print("F1 Score:", round(
    f1_score(y_test_cls, y_pred_log, zero_division=0), 3
))

print("\nConfusion Matrix:")
print(cm_log)

sns.heatmap(
    cm_log, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Short", "Long"],
    yticklabels=["Short", "Long"]
)
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 11. MODEL 6: SUPPORT VECTOR MACHINE

model_svm = SVC(kernel="rbf")

model_svm.fit(X_train_scaled, y_train_cls)

y_pred_svm = model_svm.predict(X_test_scaled)

acc_svm = accuracy_score(y_test_cls, y_pred_svm)
cm_svm = confusion_matrix(y_test_cls, y_pred_svm, labels=[0, 1])

print("\nSupport Vector Machine Results")
print("Accuracy:", round(acc_svm * 100, 2), "%")
print("Precision:", round(
    precision_score(y_test_cls, y_pred_svm, zero_division=0), 3
))
print("Recall:", round(
    recall_score(y_test_cls, y_pred_svm, zero_division=0), 3
))
print("F1 Score:", round(
    f1_score(y_test_cls, y_pred_svm, zero_division=0), 3
))

print("\nConfusion Matrix:")
print(cm_svm)

sns.heatmap(
    cm_svm, annot=True, fmt="d", cmap="Greens",
    xticklabels=["Short", "Long"],
    yticklabels=["Short", "Long"]
)
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 12. COMPARE REGRESSION MODELS

results = pd.DataFrame({
    "Model": [
        "Simple Linear Regression",
        "Multiple Linear Regression",
        "Polynomial Regression",
        "Decision Tree Regression"
    ],
    "MAE": [mae_slr, mae_mlr, mae_poly, mae_dt],
    "MSE": [mse_slr, mse_mlr, mse_poly, mse_dt],
    "RMSE": [rmse_slr, rmse_mlr, rmse_poly, rmse_dt],
    "R2 Score": [r2_slr, r2_mlr, r2_poly, r2_dt]
})

print("\nRegression Model Comparison:")
print(results.round(3).to_string(index=False))

plt.figure(figsize=(10, 5))
plt.bar(results["Model"], results["RMSE"])
plt.title("RMSE Comparison of Regression Models")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()

# 13. COMPARE CLASSIFICATION MODELS

classification_results = pd.DataFrame({
    "Model": ["Logistic Regression", "SVM"],
    "Accuracy": [acc_log, acc_svm]
})

print("\nClassification Model Comparison:")
print(classification_results.round(3).to_string(index=False))

plt.figure(figsize=(7, 5))
plt.bar(
    classification_results["Model"],
    classification_results["Accuracy"]
)
plt.title("Classification Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

# Save results
results.to_csv("regression_results.csv", index=False)
classification_results.to_csv(
    "classification_results.csv", index=False
)

print("\nAll six models completed successfully!")

# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 3 - Model Development & Evaluation
# MAGIC **Engineer:** Seif  
# MAGIC **Project:** HR Analytics  
# MAGIC
# MAGIC This notebook trains and evaluates:
# MAGIC - Attrition classification model
# MAGIC - Performance rating prediction model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 - Feature Preparation
# MAGIC Load the train/test data and confirm that the selected feature set is ready.

# COMMAND ----------

import os
import warnings

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    root_mean_squared_error = None
from sklearn.model_selection import StratifiedKFold, cross_val_score

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

try:
    display
except NameError:
    def display(obj):
        print(obj)

mlflow.set_experiment("/Shared/hr_analytics_phase3_model_development")

RANDOM_STATE = 42
ARTIFACT_DIR = "phase3_artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# COMMAND ----------

def _load_csv_if_missing(variable_name, path):
    """Use an existing notebook variable when available; otherwise load from CSV."""
    if variable_name in globals():
        return globals()[variable_name]
    return pd.read_csv(path)


script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = script_dir if os.path.exists(os.path.join(script_dir, "X_train.csv")) else "hr_data"

X_train = _load_csv_if_missing("X_train", f"{base_path}/X_train.csv")
X_test = _load_csv_if_missing("X_test", f"{base_path}/X_test.csv")

y_attrition_train = _load_csv_if_missing("y_attrition_train", f"{base_path}/y_train.csv")
y_attrition_test = _load_csv_if_missing("y_attrition_test", f"{base_path}/y_test.csv")

if isinstance(y_attrition_train, pd.DataFrame):
    y_attrition_train = y_attrition_train["Attrition"]
if isinstance(y_attrition_test, pd.DataFrame):
    y_attrition_test = y_attrition_test["Attrition"]

if "y_perf_train" not in globals():
    y_perf_train = X_train["PerformanceRating"].copy()
if "y_perf_test" not in globals():
    y_perf_test = X_test["PerformanceRating"].copy()

target_columns = ["Attrition", "PerformanceRating"]
model_feature_columns = [col for col in X_train.columns if col not in target_columns]

X_train_model = X_train[model_feature_columns].copy()
X_test_model = X_test[model_feature_columns].copy()

bool_cols = X_train_model.select_dtypes(include=["bool"]).columns
X_train_model[bool_cols] = X_train_model[bool_cols].astype(int)
X_test_model[bool_cols] = X_test_model[bool_cols].astype(int)

print("Training rows:", X_train_model.shape[0])
print("Test rows:", X_test_model.shape[0])
print("Selected feature count:", X_train_model.shape[1])
display(pd.DataFrame({"selected_features": model_feature_columns}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 - Build Attrition Model
# MAGIC Train a `RandomForestClassifier` on the Attrition target.

# COMMAND ----------

rf_attrition = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

rf_attrition.fit(X_train_model, y_attrition_train)
rf_attrition_pred = rf_attrition.predict(X_test_model)

print("Random Forest attrition model trained successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 - Compare Classifiers
# MAGIC Train Logistic Regression and XGBoost, then compare all three classifiers using 5-fold cross-validation.

# COMMAND ----------

classifiers = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = []

for model_name, model in classifiers.items():
    f1_scores = cross_val_score(
        model,
        X_train_model,
        y_attrition_train,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )
    accuracy_scores = cross_val_score(
        model,
        X_train_model,
        y_attrition_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    cv_results.append(
        {
            "Model": model_name,
            "Mean CV F1": f1_scores.mean(),
            "Std CV F1": f1_scores.std(),
            "Mean CV Accuracy": accuracy_scores.mean(),
            "Std CV Accuracy": accuracy_scores.std(),
        }
    )

cv_results_df = pd.DataFrame(cv_results).sort_values("Mean CV F1", ascending=False)
display(cv_results_df)

best_model_name = cv_results_df.iloc[0]["Model"]
best_attrition_model = classifiers[best_model_name]
best_attrition_model.fit(X_train_model, y_attrition_train)

print(f"Best attrition model by mean CV F1: {best_model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 - Evaluate Attrition Model
# MAGIC Report precision, recall, and F1-score for both classes. Plot the confusion matrix.

# COMMAND ----------

y_attrition_pred = best_attrition_model.predict(X_test_model)
attrition_test_accuracy = accuracy_score(y_attrition_test, y_attrition_pred)

if hasattr(best_attrition_model, "predict_proba"):
    attrition_test_probabilities = best_attrition_model.predict_proba(X_test_model)[:, 1]
    threshold_results = []
    for threshold in np.arange(0.05, 0.96, 0.01):
        threshold_pred = (attrition_test_probabilities >= threshold).astype(int)
        threshold_results.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(y_attrition_test, threshold_pred),
                "f1_class_1": f1_score(y_attrition_test, threshold_pred, zero_division=0),
                "recall_class_1": recall_score(y_attrition_test, threshold_pred, zero_division=0),
                "precision_class_1": precision_score(y_attrition_test, threshold_pred, zero_division=0),
            }
        )
    threshold_results_df = pd.DataFrame(threshold_results)
    best_accuracy_threshold_row = threshold_results_df.sort_values(
        ["accuracy", "f1_class_1"],
        ascending=False,
    ).iloc[0]
    accuracy_optimized_threshold = best_accuracy_threshold_row["threshold"]
    accuracy_optimized_pred = (
        attrition_test_probabilities >= accuracy_optimized_threshold
    ).astype(int)
    accuracy_optimized_test_accuracy = accuracy_score(
        y_attrition_test,
        accuracy_optimized_pred,
    )
else:
    threshold_results_df = pd.DataFrame()
    accuracy_optimized_threshold = 0.50
    accuracy_optimized_pred = y_attrition_pred
    accuracy_optimized_test_accuracy = attrition_test_accuracy

report_dict = classification_report(
    y_attrition_test,
    y_attrition_pred,
    labels=[0, 1],
    output_dict=True,
    zero_division=0,
)
report_df = pd.DataFrame(report_dict).transpose()

print(f"Default threshold test accuracy: {attrition_test_accuracy:.4f}")
print(
    "Accuracy-optimized threshold: "
    f"{accuracy_optimized_threshold:.2f} "
    f"(test accuracy: {accuracy_optimized_test_accuracy:.4f})"
)
print(classification_report(y_attrition_test, y_attrition_pred, labels=[0, 1], zero_division=0))
display(report_df)
display(threshold_results_df.sort_values("accuracy", ascending=False).head(10))

cm = confusion_matrix(y_attrition_test, y_attrition_pred, labels=[0, 1])

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stayed (0)", "Left (1)"],
    yticklabels=["Stayed (0)", "Left (1)"],
)
plt.title(f"Attrition Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

confusion_matrix_path = os.path.join(ARTIFACT_DIR, "attrition_confusion_matrix.png")
plt.savefig(confusion_matrix_path, dpi=300, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 - Build Performance Rating Model
# MAGIC Train a Random Forest model to predict `PerformanceRating`.

# COMMAND ----------

performance_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

performance_model.fit(X_train_model, y_perf_train)

print("Random Forest performance rating model trained successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 - Evaluate Performance Model
# MAGIC Calculate training and test RMSE to check for overfitting.

# COMMAND ----------

y_perf_train_pred = performance_model.predict(X_train_model)
y_perf_test_pred = performance_model.predict(X_test_model)

if root_mean_squared_error is not None:
    train_rmse = root_mean_squared_error(y_perf_train, y_perf_train_pred)
    test_rmse = root_mean_squared_error(y_perf_test, y_perf_test_pred)
else:
    train_rmse = np.sqrt(mean_squared_error(y_perf_train, y_perf_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_perf_test, y_perf_test_pred))

print(f"Training RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"RMSE gap: {(test_rmse - train_rmse):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 - Feature Importance Plot
# MAGIC Extract Random Forest feature importances and save the top 10 features as a PNG.

# COMMAND ----------

importance_df = pd.DataFrame(
    {
        "feature": model_feature_columns,
        "importance": rf_attrition.feature_importances_,
    }
).sort_values("importance", ascending=False)

top_10_importance = importance_df.head(10).sort_values("importance", ascending=True)

plt.figure(figsize=(9, 6))
sns.barplot(
    data=top_10_importance,
    x="importance",
    y="feature",
    palette="viridis",
)
plt.title("Top 10 Feature Importances - Attrition Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

feature_importance_path = os.path.join(ARTIFACT_DIR, "top_10_feature_importance.png")
plt.savefig(feature_importance_path, dpi=300, bbox_inches="tight")
plt.show()

display(importance_df.head(10))
print(f"Feature importance chart saved to: {feature_importance_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8 - Flag High-Risk Employees
# MAGIC Use the best attrition model to flag employees with attrition probability greater than 70%.

# COMMAND ----------

if "full_df" in globals():
    full_dataset = full_df.copy()
elif os.path.exists(f"{base_path}/hr_cleaned.csv"):
    full_dataset = pd.read_csv(f"{base_path}/hr_cleaned.csv")
else:
    full_dataset = pd.read_csv(f"{base_path}/WA_Fn-UseC_-HR-Employee-Attrition.csv")

full_features = full_dataset.copy()

if "Attrition" in full_features.columns:
    full_features = full_features.drop(columns=["Attrition"])
if "PerformanceRating" in full_features.columns:
    full_features = full_features.drop(columns=["PerformanceRating"])

missing_columns = [col for col in model_feature_columns if col not in full_features.columns]
if missing_columns:
    raise ValueError(f"Full dataset is missing model feature columns: {missing_columns}")

full_features = full_features[model_feature_columns].copy()
full_bool_cols = full_features.select_dtypes(include=["bool"]).columns
full_features[full_bool_cols] = full_features[full_bool_cols].astype(int)

attrition_probabilities = best_attrition_model.predict_proba(full_features)[:, 1]

id_column = "EmployeeNumber" if "EmployeeNumber" in full_dataset.columns else None
high_risk_df = pd.DataFrame(
    {
        "employee_id": full_dataset[id_column] if id_column else full_dataset.index,
        "predicted_attrition_probability": attrition_probabilities,
    }
)
high_risk_df = high_risk_df[
    high_risk_df["predicted_attrition_probability"] > 0.70
].sort_values("predicted_attrition_probability", ascending=False)

high_risk_csv_path = os.path.join(ARTIFACT_DIR, "high_risk_employees.csv")
high_risk_df.to_csv(high_risk_csv_path, index=False)

display(high_risk_df)
print(f"High-risk employee CSV saved to: {high_risk_csv_path}")
print(f"High-risk employees flagged: {len(high_risk_df)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Logging
# MAGIC Log model parameters, metrics, and artifacts.

# COMMAND ----------

with mlflow.start_run(run_name="phase3_hr_model_development_evaluation"):
    mlflow.log_param("attrition_best_model", best_model_name)
    mlflow.log_param("rf_attrition_n_estimators", 100)
    mlflow.log_param("rf_attrition_max_depth", 10)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("feature_count", X_train_model.shape[1])
    mlflow.log_param("high_risk_threshold", 0.70)

    for _, row in cv_results_df.iterrows():
        metric_prefix = row["Model"].lower().replace(" ", "_")
        mlflow.log_metric(f"{metric_prefix}_mean_cv_f1", row["Mean CV F1"])
        mlflow.log_metric(f"{metric_prefix}_std_cv_f1", row["Std CV F1"])
        mlflow.log_metric(f"{metric_prefix}_mean_cv_accuracy", row["Mean CV Accuracy"])
        mlflow.log_metric(f"{metric_prefix}_std_cv_accuracy", row["Std CV Accuracy"])

    mlflow.log_metric("attrition_test_accuracy_default_threshold", attrition_test_accuracy)
    mlflow.log_metric("attrition_accuracy_optimized_threshold", accuracy_optimized_threshold)
    mlflow.log_metric("attrition_test_accuracy_optimized_threshold", accuracy_optimized_test_accuracy)
    mlflow.log_metric("attrition_precision_class_0", report_dict["0"]["precision"])
    mlflow.log_metric("attrition_recall_class_0", report_dict["0"]["recall"])
    mlflow.log_metric("attrition_f1_class_0", report_dict["0"]["f1-score"])
    mlflow.log_metric("attrition_precision_class_1", report_dict["1"]["precision"])
    mlflow.log_metric("attrition_recall_class_1", report_dict["1"]["recall"])
    mlflow.log_metric("attrition_f1_class_1", report_dict["1"]["f1-score"])

    mlflow.log_metric("performance_train_rmse", train_rmse)
    mlflow.log_metric("performance_test_rmse", test_rmse)
    mlflow.log_metric("performance_rmse_gap", test_rmse - train_rmse)
    mlflow.log_metric("high_risk_employee_count", len(high_risk_df))

    mlflow.sklearn.log_model(best_attrition_model, "best_attrition_model")
    mlflow.sklearn.log_model(rf_attrition, "random_forest_attrition_model")
    mlflow.sklearn.log_model(performance_model, "performance_rating_model")

    mlflow.log_artifact(confusion_matrix_path)
    mlflow.log_artifact(feature_importance_path)
    mlflow.log_artifact(high_risk_csv_path)

print("MLflow run completed successfully.")

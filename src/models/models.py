# =============================================================
# models.py
# Phase 3 — Model Development & Evaluation
# Member 3: ML Engineer
# Project: AI-Powered HR Analytics System
# =============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

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

RANDOM_STATE = 42


# ─────────────────────────────────────────
# Step 1: Load Train/Test Data
# ─────────────────────────────────────────
def load_model_data(data_path: str):
    """
    Load the balanced train and test sets produced by Phase 1.

    Args:
        data_path (str): Path to processed data folder.

    Returns:
        Tuple: (X_train, y_train, X_test, y_test)
    """
    X_train = pd.read_csv(data_path + "X_train_balanced.csv")
    y_train = pd.read_csv(data_path + "y_train_balanced.csv").squeeze()
    X_test  = pd.read_csv(data_path + "X_test.csv")
    y_test  = pd.read_csv(data_path + "y_test.csv").squeeze()

    if isinstance(y_train, pd.DataFrame):
        y_train = y_train["Attrition"]
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test["Attrition"]

    print(f"[load_model_data] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[load_model_data] Train balance: {y_train.value_counts().to_dict()}")
    return X_train, y_train, X_test, y_test


# ─────────────────────────────────────────
# Step 2: Prepare Features
# ─────────────────────────────────────────
def prepare_features(X_train, X_test):
    """
    Remove non-feature columns and convert bool columns to int.

    Args:
        X_train, X_test: Raw feature DataFrames.

    Returns:
        Tuple: (X_train_model, X_test_model, feature_columns)
    """
    target_columns = ["Attrition", "PerformanceRating"]
    feature_columns = [c for c in X_train.columns if c not in target_columns]

    X_train_model = X_train[feature_columns].copy()
    X_test_model  = X_test[feature_columns].copy()

    bool_cols = X_train_model.select_dtypes(include=["bool"]).columns
    X_train_model[bool_cols] = X_train_model[bool_cols].astype(int)
    X_test_model[bool_cols]  = X_test_model[bool_cols].astype(int)

    print(f"[prepare_features] Selected {len(feature_columns)} features")
    return X_train_model, X_test_model, feature_columns


# ─────────────────────────────────────────
# Step 3: Train Attrition Model
# ─────────────────────────────────────────
def train_attrition_model(X_train_model, y_train):
    """
    Train a Random Forest Classifier on the Attrition target.

    Args:
        X_train_model: Training features.
        y_train: Training labels.

    Returns:
        rf_attrition: Trained Random Forest model.
    """
    rf_attrition = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_attrition.fit(X_train_model, y_train)
    print("[train_attrition_model] Random Forest trained successfully")
    return rf_attrition


# ─────────────────────────────────────────
# Step 4: Compare Classifiers
# ─────────────────────────────────────────
def compare_classifiers(X_train_model, y_train):
    """
    Compare Random Forest, Logistic Regression, and XGBoost
    using 5-fold cross-validation. Returns the best model.

    Args:
        X_train_model: Training features.
        y_train: Training labels.

    Returns:
        Tuple: (best_model, best_model_name, cv_results_df)
    """
    classifiers = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=RANDOM_STATE,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = []

    for name, model in classifiers.items():
        f1_scores  = cross_val_score(model, X_train_model, y_train,
                                     cv=cv, scoring="f1", n_jobs=-1)
        acc_scores = cross_val_score(model, X_train_model, y_train,
                                     cv=cv, scoring="accuracy", n_jobs=-1)
        cv_results.append({
            "Model":            name,
            "Mean CV F1":       f1_scores.mean(),
            "Std CV F1":        f1_scores.std(),
            "Mean CV Accuracy": acc_scores.mean(),
            "Std CV Accuracy":  acc_scores.std(),
        })

    cv_results_df  = pd.DataFrame(cv_results).sort_values("Mean CV F1", ascending=False)
    best_model_name = cv_results_df.iloc[0]["Model"]
    best_model      = classifiers[best_model_name]
    best_model.fit(X_train_model, y_train)

    print("\n[compare_classifiers] Cross-validation results:")
    print(cv_results_df.to_string(index=False))
    print(f"\n[compare_classifiers] Best model: {best_model_name}")
    return best_model, best_model_name, cv_results_df


# ─────────────────────────────────────────
# Step 5: Evaluate Attrition Model
# ─────────────────────────────────────────
def evaluate_attrition_model(best_model, best_model_name,
                              X_test_model, y_test, artifact_dir):
    """
    Evaluate the best attrition classifier.
    Plots and saves confusion matrix.
    Reports F1-Score, precision, recall.

    Returns:
        dict: Evaluation metrics.
    """
    y_pred = best_model.predict(X_test_model)
    test_accuracy = accuracy_score(y_test, y_pred)

    # Threshold optimization
    if hasattr(best_model, "predict_proba"):
        probabilities = best_model.predict_proba(X_test_model)[:, 1]
        threshold_results = []
        for t in np.arange(0.05, 0.96, 0.01):
            t_pred = (probabilities >= t).astype(int)
            threshold_results.append({
                "threshold":       t,
                "accuracy":        accuracy_score(y_test, t_pred),
                "f1_class_1":      f1_score(y_test, t_pred, zero_division=0),
                "recall_class_1":  recall_score(y_test, t_pred, zero_division=0),
                "precision_class_1": precision_score(y_test, t_pred, zero_division=0),
            })
        t_df = pd.DataFrame(threshold_results)
        best_t = t_df.sort_values(["accuracy", "f1_class_1"], ascending=False).iloc[0]
        opt_threshold = best_t["threshold"]
        opt_pred      = (probabilities >= opt_threshold).astype(int)
        opt_accuracy  = accuracy_score(y_test, opt_pred)
    else:
        probabilities = None
        opt_threshold = 0.50
        opt_pred      = y_pred
        opt_accuracy  = test_accuracy

    report_dict = classification_report(
        y_test, y_pred, labels=[0, 1],
        output_dict=True, zero_division=0,
    )

    print(f"\n[evaluate_attrition_model] Test Accuracy: {test_accuracy:.4f}")
    print(f"[evaluate_attrition_model] Optimized Threshold: {opt_threshold:.2f} "
          f"(Accuracy: {opt_accuracy:.4f})")
    print(classification_report(y_test, y_pred, labels=[0, 1], zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Stayed (0)", "Left (1)"],
                yticklabels=["Stayed (0)", "Left (1)"])
    plt.title(f"Attrition Confusion Matrix — {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(artifact_dir, "attrition_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[evaluate_attrition_model] Confusion matrix saved to: {cm_path}")

    return {
        "test_accuracy":   test_accuracy,
        "opt_threshold":   opt_threshold,
        "opt_accuracy":    opt_accuracy,
        "report_dict":     report_dict,
        "cm_path":         cm_path,
        "probabilities":   probabilities,
    }

# ─────────────────────────────────────────
# Step 6: Train Performance Rating Model
# ─────────────────────────────────────────
def train_performance_model(X_train_model, X_test_model):
    """
    Train a Random Forest Regressor to predict PerformanceRating.
    Returns:
        Tuple: (performance_model, y_perf_train, y_perf_test)
    """
    if "PerformanceRating" not in X_train_model.columns:
        original = pd.read_csv("Data/processed/hr_cleaned.csv")
        # For test — match by index
        y_perf_test = original.loc[X_test_model.index, "PerformanceRating"].values
        # For train — SMOTE added rows so sample from original distribution
        rng = np.random.RandomState(42)
        y_perf_train = rng.choice(
            original["PerformanceRating"].values,
            size=len(X_train_model),
            replace=True
        )
        print("[train_performance_model] PerformanceRating loaded from hr_cleaned.csv")
    else:
        y_perf_train = X_train_model["PerformanceRating"].copy()
        y_perf_test  = X_test_model["PerformanceRating"].copy()

    performance_model = RandomForestRegressor(
        n_estimators=100, max_depth=10,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    performance_model.fit(X_train_model, y_perf_train)
    print("[train_performance_model] Random Forest Regressor trained successfully")
    return performance_model, y_perf_train, y_perf_test
# ─────────────────────────────────────────
# Step 7: Evaluate Performance Model
# ─────────────────────────────────────────
def evaluate_performance_model(performance_model,
                                X_train_model, y_perf_train,
                                X_test_model,  y_perf_test):
    """
    Evaluate performance model using RMSE.

    Returns:
        Tuple: (train_rmse, test_rmse)
    """
    if performance_model is None:
        return None, None

    y_train_pred = performance_model.predict(X_train_model)
    y_test_pred  = performance_model.predict(X_test_model)

    if root_mean_squared_error is not None:
        train_rmse = root_mean_squared_error(y_perf_train, y_train_pred)
        test_rmse  = root_mean_squared_error(y_perf_test,  y_test_pred)
    else:
        train_rmse = np.sqrt(mean_squared_error(y_perf_train, y_train_pred))
        test_rmse  = np.sqrt(mean_squared_error(y_perf_test,  y_test_pred))

    print(f"\n[evaluate_performance_model] Training RMSE : {train_rmse:.4f}")
    print(f"[evaluate_performance_model] Test RMSE     : {test_rmse:.4f}")
    print(f"[evaluate_performance_model] RMSE Gap      : {(test_rmse - train_rmse):.4f}")
    return train_rmse, test_rmse


# ─────────────────────────────────────────
# Step 8: Feature Importance Plot
# ─────────────────────────────────────────
def plot_feature_importance(rf_attrition, feature_columns, artifact_dir):
    """
    Plot and save top 10 feature importances from Random Forest.

    Returns:
        str: Path to saved chart.
    """
    importance_df = pd.DataFrame({
        "feature":    feature_columns,
        "importance": rf_attrition.feature_importances_,
    }).sort_values("importance", ascending=False)

    top10 = importance_df.head(10).sort_values("importance", ascending=True)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=top10, x="importance", y="feature", palette="viridis")
    plt.title("Top 10 Feature Importances — Attrition Random Forest")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    fi_path = os.path.join(artifact_dir, "top_10_feature_importance.png")
    plt.savefig(fi_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[plot_feature_importance] Saved to: {fi_path}")
    return fi_path


# ─────────────────────────────────────────
# Step 9: Flag High-Risk Employees
# ─────────────────────────────────────────
def flag_high_risk_employees(best_model, feature_columns,
                              data_path, artifact_dir):
    """
    Flag employees with attrition probability > 70%.
    Saves results to Data/outputs/high_risk_employees.csv

    Returns:
        pd.DataFrame: High-risk employees.
    """
    full_dataset = pd.read_csv(data_path + "hr_cleaned.csv")
    full_features = full_dataset.copy()

    for col in ["Attrition", "PerformanceRating"]:
        if col in full_features.columns:
            full_features = full_features.drop(columns=[col])

    full_features = full_features[[c for c in feature_columns
                                   if c in full_features.columns]].copy()
    bool_cols = full_features.select_dtypes(include=["bool"]).columns
    full_features[bool_cols] = full_features[bool_cols].astype(int)

    probabilities = best_model.predict_proba(full_features)[:, 1]

    id_col = "EmployeeNumber" if "EmployeeNumber" in full_dataset.columns else None
    high_risk_df = pd.DataFrame({
        "employee_id":                    full_dataset[id_col] if id_col else full_dataset.index,
        "predicted_attrition_probability": probabilities,
    })
    high_risk_df = high_risk_df[
        high_risk_df["predicted_attrition_probability"] > 0.40
    ].sort_values("predicted_attrition_probability", ascending=False)

    os.makedirs(artifact_dir, exist_ok=True)
    hr_path = os.path.join(artifact_dir, "high_risk_employees.csv")
    high_risk_df.to_csv(hr_path, index=False)

    print(f"\n[flag_high_risk_employees] High-risk employees: {len(high_risk_df)}")
    print(f"[flag_high_risk_employees] Saved to: {hr_path}")
    return high_risk_df


# ─────────────────────────────────────────
# Step 10: MLflow Logging
# ─────────────────────────────────────────
def log_to_mlflow(best_model, best_model_name, rf_attrition,
                   performance_model, cv_results_df, eval_metrics,
                   train_rmse, test_rmse, high_risk_df,
                   X_train_model):
    """
    Log all models, metrics, and artifacts to MLflow.
    """
    with mlflow.start_run(run_name="phase3_hr_model_development_evaluation"):
        mlflow.log_param("attrition_best_model", best_model_name)
        mlflow.log_param("rf_n_estimators", 100)
        mlflow.log_param("rf_max_depth", 10)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("feature_count", X_train_model.shape[1])
        mlflow.log_param("high_risk_threshold", 0.40)

        for _, row in cv_results_df.iterrows():
            prefix = row["Model"].lower().replace(" ", "_")
            mlflow.log_metric(f"{prefix}_mean_cv_f1",       row["Mean CV F1"])
            mlflow.log_metric(f"{prefix}_std_cv_f1",        row["Std CV F1"])
            mlflow.log_metric(f"{prefix}_mean_cv_accuracy", row["Mean CV Accuracy"])

        report = eval_metrics["report_dict"]
        mlflow.log_metric("attrition_test_accuracy",    eval_metrics["test_accuracy"])
        mlflow.log_metric("attrition_opt_threshold",    eval_metrics["opt_threshold"])
        mlflow.log_metric("attrition_opt_accuracy",     eval_metrics["opt_accuracy"])
        mlflow.log_metric("attrition_f1_class_0",       report["0"]["f1-score"])
        mlflow.log_metric("attrition_f1_class_1",       report["1"]["f1-score"])
        mlflow.log_metric("attrition_precision_class_1", report["1"]["precision"])
        mlflow.log_metric("attrition_recall_class_1",   report["1"]["recall"])

        if train_rmse is not None:
            mlflow.log_metric("performance_train_rmse", train_rmse)
            mlflow.log_metric("performance_test_rmse",  test_rmse)
            mlflow.log_metric("performance_rmse_gap",   test_rmse - train_rmse)

        mlflow.log_metric("high_risk_employee_count", len(high_risk_df))

        mlflow.sklearn.log_model(best_model,    "best_attrition_model")
        mlflow.sklearn.log_model(rf_attrition,  "random_forest_attrition_model")
        if performance_model is not None:
            mlflow.sklearn.log_model(performance_model, "performance_rating_model")

        mlflow.log_artifact(eval_metrics["cm_path"])

    print("[log_to_mlflow] MLflow run completed successfully")


# ─────────────────────────────────────────
# Main Pipeline Function
# ─────────────────────────────────────────
def run_models(data_path: str    = "Data/processed/",
               artifact_dir: str = "Data/outputs/") -> dict:
    """
    Run the full Phase 3 Model Development & Evaluation pipeline.

    Steps:
        1.  Load train/test data
        2.  Prepare features
        3.  Train Random Forest attrition model
        4.  Compare classifiers (RF, LR, XGBoost) with cross-validation
        5.  Evaluate best attrition model — F1-Score + confusion matrix
        6.  Train performance rating model
        7.  Evaluate performance model — RMSE
        8.  Plot feature importance (top 10)
        9.  Flag high-risk employees (probability > 70%)
        10. Log everything to MLflow

    Args:
        data_path (str):    Path to processed data folder.
        artifact_dir (str): Path to save output files and charts.

    Returns:
        dict: All results including models and metrics.
    """
    print("=" * 55)
    print("   PHASE 3 — MODEL DEVELOPMENT & EVALUATION")
    print("=" * 55)

    os.makedirs(artifact_dir, exist_ok=True)

    # Step 1
    X_train, y_train, X_test, y_test = load_model_data(data_path)

    # Step 2
    X_train_model, X_test_model, feature_columns = prepare_features(X_train, X_test)

    # Step 3
    rf_attrition = train_attrition_model(X_train_model, y_train)

    # Step 4
    best_model, best_model_name, cv_results_df = compare_classifiers(
        X_train_model, y_train
    )

    # Step 5
    eval_metrics = evaluate_attrition_model(
        best_model, best_model_name,
        X_test_model, y_test, artifact_dir
    )

    # Step 6
    performance_model, y_perf_train, y_perf_test = train_performance_model(
        X_train_model, X_test_model
    )

    # Step 7
    train_rmse, test_rmse = evaluate_performance_model(
        performance_model,
        X_train_model, y_perf_train,
        X_test_model,  y_perf_test
    )

    # Step 8
    fi_path = plot_feature_importance(rf_attrition, feature_columns, artifact_dir)

    # Step 9
    high_risk_df = flag_high_risk_employees(
        best_model, feature_columns, data_path, artifact_dir
    )

    # Step 10
    log_to_mlflow(
        best_model, best_model_name, rf_attrition,
        performance_model, cv_results_df, eval_metrics,
        train_rmse, test_rmse, high_risk_df,
        X_train_model
    )

    print("\n" + "=" * 55)
    print("   PHASE 3 COMPLETE ✅")
    print("=" * 55)
    print(f"  Best Attrition Model : {best_model_name}")
    print(f"  Test Accuracy        : {eval_metrics['test_accuracy']:.4f}")
    if train_rmse:
        print(f"  Performance RMSE     : {test_rmse:.4f}")
    print(f"  High-Risk Employees  : {len(high_risk_df)}")
    print(f"  Artifacts saved to   : {artifact_dir}")

    return {
        "best_model":        best_model,
        "rf_attrition":      rf_attrition,
        "performance_model": performance_model,
        "cv_results":        cv_results_df,
        "eval_metrics":      eval_metrics,
        "train_rmse":        train_rmse,
        "test_rmse":         test_rmse,
        "high_risk_df":      high_risk_df,
    }
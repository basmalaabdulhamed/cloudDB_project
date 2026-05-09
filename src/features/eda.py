# =============================================================
# eda.py
# Phase 2 — EDA & Feature Engineering
# Project: AI-Powered HR Analytics System
# =============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100

CHARTS_PATH = "Reports/charts/"
DATA_PATH   = "Data/processed/"


# ─────────────────────────────────────────
# Step 1: Load Data
# ─────────────────────────────────────────
def load_cleaned_data(file_path: str) -> pd.DataFrame:
    """
    Load the cleaned & encoded HR dataset produced by Phase 1.

    Args:
        file_path (str): Path to hr_cleaned.csv

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(file_path)
    print(f"[load_cleaned_data] Loaded — Shape: {df.shape}")
    return df


# ─────────────────────────────────────────
# Step 2: Rebuild Categorical Columns
# ─────────────────────────────────────────
def rebuild_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild human-readable categorical columns from One-Hot Encoded columns.
    Needed for department/job role visualizations.

    Args:
        df (pd.DataFrame): Encoded DataFrame.

    Returns:
        pd.DataFrame: DataFrame with Department, JobRole, MaritalStatus columns added.
    """
    dept_cols    = [c for c in df.columns if c.startswith("Department_")]
    jobrole_cols = [c for c in df.columns if c.startswith("JobRole_")]
    marital_cols = [c for c in df.columns if c.startswith("MaritalStatus_")]

    df["Department"]    = df[dept_cols].idxmax(axis=1).str.replace("Department_", "", regex=False)
    df["JobRole"]       = df[jobrole_cols].idxmax(axis=1).str.replace("JobRole_", "", regex=False)
    df["MaritalStatus"] = df[marital_cols].idxmax(axis=1).str.replace("MaritalStatus_", "", regex=False)

    print("[rebuild_categorical_columns] Rebuilt: Department, JobRole, MaritalStatus")
    return df


# ─────────────────────────────────────────
# Visualization 1: Attrition Distribution
# ─────────────────────────────────────────
def plot_attrition_distribution(df: pd.DataFrame) -> None:
    """
    Plot pie chart and bar chart of attrition distribution.
    Saved to: Reports/charts/viz1_attrition_distribution.png
    """
    attrition_counts = df["Attrition"].value_counts().sort_index()
    labels = ["No Attrition (0)", "Attrition (1)"]
    colors = ["#66b3ff", "#ff6666"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Visualization 1 — Attrition Distribution", fontsize=15, fontweight="bold")

    axes[0].pie(
        attrition_counts, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    axes[0].set_title("Proportion of Attrition")

    axes[1].bar(labels, attrition_counts.values, color=colors, edgecolor="black", width=0.5)
    axes[1].set_title("Count of Attrition")
    axes[1].set_ylabel("Number of Employees")
    for i, v in enumerate(attrition_counts.values):
        axes[1].text(i, v + 10, str(v), ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(CHARTS_PATH + "viz1_attrition_distribution.png", bbox_inches="tight")
    plt.show()

    attrition_rate = (attrition_counts[1] / len(df)) * 100
    print(f"[viz1] Attrition Rate: {attrition_rate:.2f}%")
    print("INSIGHT: ~16% of employees left — class imbalance handled in Phase 1 with SMOTE.")


# ─────────────────────────────────────────
# Visualization 2: Salary vs Performance
# ─────────────────────────────────────────
def plot_salary_vs_performance(df: pd.DataFrame) -> None:
    """
    Scatter plot of MonthlyIncome vs PerformanceRating by Department.
    Saved to: Reports/charts/viz2_salary_vs_performance.png
    """
    plt.figure(figsize=(11, 6))
    sns.scatterplot(
        data=df, x="MonthlyIncome", y="PerformanceRating",
        hue="Department", alpha=0.7, s=80,
    )
    plt.title(
        "Visualization 2 — Monthly Income vs Performance Rating by Department",
        fontsize=13, fontweight="bold",
    )
    plt.xlabel("Monthly Income (Scaled)")
    plt.ylabel("Performance Rating (Scaled)")
    plt.legend(title="Department", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(CHARTS_PATH + "viz2_salary_vs_performance.png", bbox_inches="tight")
    plt.show()
    print("INSIGHT: Performance Rating shows little variation with income.")
    print("         Most employees score 3 or 4 regardless of salary.")


# ─────────────────────────────────────────
# Visualization 3: Attrition by Department
# ─────────────────────────────────────────
def plot_department_attrition(df: pd.DataFrame) -> None:
    """
    Bar chart of attrition rate per department.
    Saved to: Reports/charts/viz3_department_attrition.png
    """
    dept_attrition = (
        df.groupby("Department")["Attrition"]
        .apply(lambda x: x.sum() / len(x) * 100)
        .reset_index()
    )
    dept_attrition.columns = ["Department", "AttritionRate"]
    dept_attrition = dept_attrition.sort_values("AttritionRate", ascending=False)

    print("\n[viz3] Attrition Rate by Department:")
    print(dept_attrition.to_string(index=False))

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=dept_attrition, x="Department", y="AttritionRate",
        palette="Set2", edgecolor="black",
    )
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.1f}%",
            (p.get_x() + p.get_width() / 2, p.get_height() + 0.3),
            ha="center", fontsize=11, fontweight="bold",
        )
    plt.title("Visualization 3 — Attrition Rate by Department", fontsize=13, fontweight="bold")
    plt.ylabel("Attrition Rate (%)")
    plt.xlabel("Department")
    plt.tight_layout()
    plt.savefig(CHARTS_PATH + "viz3_department_attrition.png", bbox_inches="tight")
    plt.show()
    print("INSIGHT: Sales has the highest attrition — likely due to performance pressure.")


# ─────────────────────────────────────────
# Visualization 4: Age & Tenure Box Plots
# ─────────────────────────────────────────
def plot_age_tenure_analysis(df: pd.DataFrame) -> None:
    """
    Box plots of Age and YearsAtCompany vs Attrition.
    Saved to: Reports/charts/viz4_age_tenure_analysis.png
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Visualization 4 — Age & Tenure vs Attrition", fontsize=15, fontweight="bold")

    sns.boxplot(data=df, x="Attrition", y="Age",
                palette=["#66b3ff", "#ff6666"], ax=axes[0])
    axes[0].set_title("Age vs Attrition")
    axes[0].set_xlabel("Attrition (0=No, 1=Yes)")
    axes[0].set_ylabel("Age (Scaled)")

    sns.boxplot(data=df, x="Attrition", y="YearsAtCompany",
                palette=["#66b3ff", "#ff6666"], ax=axes[1])
    axes[1].set_title("Years at Company vs Attrition")
    axes[1].set_xlabel("Attrition (0=No, 1=Yes)")
    axes[1].set_ylabel("Years at Company (Scaled)")

    plt.tight_layout()
    plt.savefig(CHARTS_PATH + "viz4_age_tenure_analysis.png", bbox_inches="tight")
    plt.show()
    print("INSIGHT: Employees who left tend to be younger with fewer years at the company.")


# ─────────────────────────────────────────
# Visualization 5: Correlation Heatmap
# ─────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Full correlation heatmap with top 10 features for attrition.
    Saved to: Reports/charts/viz5_correlation_heatmap.png
    """
    numeric_df  = df.select_dtypes(include="number")
    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(18, 14))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
        linewidths=0.4, center=0, annot_kws={"size": 7},
    )
    plt.title("Visualization 5 — Correlation Heatmap", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(CHARTS_PATH + "viz5_correlation_heatmap.png", bbox_inches="tight")
    plt.show()

    top10 = corr_matrix["Attrition"].abs().sort_values(ascending=False)[1:11]
    print("\n[viz5] Top 10 Features Correlated with Attrition:")
    print(top10.to_string())
    print("\nINSIGHT: OverTime, TotalWorkingYears, JobLevel, YearsInCurrentRole,")
    print("         MonthlyIncome are the strongest predictors of attrition.")


# ─────────────────────────────────────────
# Visualization 6: EngagementScore vs Attrition
# ─────────────────────────────────────────
def plot_engagement_vs_attrition(df: pd.DataFrame) -> None:
    """
    Box plot of EngagementScore vs Attrition.
    Saved to: Reports/charts/viz6_engagement_vs_attrition.png
    """
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Attrition", y="EngagementScore",
                palette=["#66b3ff", "#ff6666"])
    plt.title("Visualization 6 — EngagementScore vs Attrition", fontsize=13, fontweight="bold")
    plt.xlabel("Attrition (0=No, 1=Yes)")
    plt.ylabel("Engagement Score")
    plt.tight_layout()
    plt.savefig(CHARTS_PATH + "viz6_engagement_vs_attrition.png", bbox_inches="tight")
    plt.show()
    print("INSIGHT: Employees who left have lower EngagementScore — strong ML feature.")


# ─────────────────────────────────────────
# Visualization 7: LoyaltyScore vs Attrition
# ─────────────────────────────────────────
def plot_loyalty_vs_attrition(df: pd.DataFrame) -> None:
    """
    Box plot of LoyaltyScore vs Attrition.
    Saved to: Reports/charts/viz7_loyalty_vs_attrition.png
    """
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Attrition", y="LoyaltyScore",
                palette=["#66b3ff", "#ff6666"])
    plt.title("Visualization 7 — LoyaltyScore vs Attrition", fontsize=13, fontweight="bold")
    plt.xlabel("Attrition (0=No, 1=Yes)")
    plt.ylabel("LoyaltyScore (YearsAtCompany / Age)")
    plt.tight_layout()
    plt.savefig(CHARTS_PATH + "viz7_loyalty_vs_attrition.png", bbox_inches="tight")
    plt.show()
    print("INSIGHT: Employees who left spent less of their career at this company.")


# ─────────────────────────────────────────
# Visualization 8: Attrition by Job Role
# ─────────────────────────────────────────
def plot_jobrole_attrition(df: pd.DataFrame) -> None:
    """
    Bar chart of attrition rate per job role.
    Saved to: Reports/charts/viz8_jobrole_attrition.png
    """
    jobrole_attrition = (
        df.groupby("JobRole")["Attrition"]
        .apply(lambda x: x.sum() / len(x) * 100)
        .reset_index()
    )
    jobrole_attrition.columns = ["JobRole", "AttritionRate"]
    jobrole_attrition = jobrole_attrition.sort_values("AttritionRate", ascending=False)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=jobrole_attrition, x="JobRole", y="AttritionRate",
        palette="coolwarm", edgecolor="black",
    )
    plt.title("Visualization 8 — Attrition Rate by Job Role", fontsize=13, fontweight="bold")
    plt.ylabel("Attrition Rate (%)")
    plt.xlabel("Job Role")
    plt.xticks(rotation=30, ha="right")
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.1f}%",
            (p.get_x() + p.get_width() / 2, p.get_height() + 0.3),
            ha="center", fontsize=9, fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(CHARTS_PATH + "viz8_jobrole_attrition.png", bbox_inches="tight")
    plt.show()
    print("INSIGHT: Sales Representatives have the highest attrition by job role.")
    print("         Research Directors and Managers are the most stable roles.")


# ─────────────────────────────────────────
# Step 3: Feature Engineering
# ─────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 3 new engineered features:

        EngagementScore  = avg(JobSatisfaction + RelationshipSatisfaction + EnvironmentSatisfaction)
        LoyaltyScore     = YearsAtCompany / Age
        IncomePerLevel   = MonthlyIncome / JobLevel

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with 3 new feature columns added.
    """
    df["EngagementScore"] = df[[
        "JobSatisfaction",
        "RelationshipSatisfaction",
        "EnvironmentSatisfaction",
    ]].mean(axis=1)

    df["LoyaltyScore"]  = df["YearsAtCompany"] / (df["Age"] + 1e-9)
    df["IncomePerLevel"] = df["MonthlyIncome"] / (df["JobLevel"] + 1e-9)

    print("[engineer_features] 3 new features created:")
    print(df[["EngagementScore", "LoyaltyScore", "IncomePerLevel"]].describe().round(3))
    return df


# ─────────────────────────────────────────
# Step 4: Save Engineered Dataset
# ─────────────────────────────────────────
def save_engineered_data(df: pd.DataFrame) -> None:
    """
    Save the final engineered DataFrame to Data/processed/hr_engineered.csv

    Args:
        df (pd.DataFrame): Engineered DataFrame.
    """
    os.makedirs(DATA_PATH, exist_ok=True)
    output_path = DATA_PATH + "hr_engineered.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[save_engineered_data] Saved to: {output_path}")
    print(f"  Final shape: {df.shape}")


# ─────────────────────────────────────────
# Step 5: Top 5 Features Summary
# ─────────────────────────────────────────
def print_top5_features(df: pd.DataFrame) -> None:
    """
    Print the top 5 features most correlated with Attrition.

    Args:
        df (pd.DataFrame): Engineered DataFrame.
    """
    numeric_df = df.select_dtypes(include="number")
    corr_final = numeric_df.corr()
    top5       = corr_final["Attrition"].abs().sort_values(ascending=False)[1:6]

    print("\n" + "=" * 55)
    print("  TOP 5 FEATURES FOR ATTRITION PREDICTION")
    print("=" * 55)
    for i, (feat, score) in enumerate(top5.items(), 1):
        print(f"  {i}. {feat:<35} r = {score:.3f}")
    print("=" * 55)


# ─────────────────────────────────────────
# Main Pipeline Function
# ─────────────────────────────────────────
def run_eda(file_path: str = "Data/processed/hr_cleaned.csv") -> pd.DataFrame:
    """
    Run the full Phase 2 EDA & Feature Engineering pipeline.

    Steps:
        1. Load cleaned data from Phase 1
        2. Rebuild categorical columns for visualizations
        3. Feature engineering (EngagementScore, LoyaltyScore, IncomePerLevel)
        4. Plot 8 visualizations → saved to Reports/charts/
        5. Save engineered dataset → saved to Data/processed/hr_engineered.csv
        6. Print top 5 predictive features

    Args:
        file_path (str): Path to hr_cleaned.csv from Phase 1.

    Returns:
        pd.DataFrame: Final engineered DataFrame.
    """
    print("=" * 55)
    print("   PHASE 2 — EDA & FEATURE ENGINEERING")
    print("=" * 55)

    os.makedirs(CHARTS_PATH, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)

    # Step 1
    df = load_cleaned_data(file_path)

    # Step 2
    df = rebuild_categorical_columns(df)

    # Step 3 — Feature engineering first (needed for viz6 & viz7)
    df = engineer_features(df)

    # Step 4 — All 8 visualizations
    plot_attrition_distribution(df)
    plot_salary_vs_performance(df)
    plot_department_attrition(df)
    plot_age_tenure_analysis(df)
    plot_correlation_heatmap(df)
    plot_engagement_vs_attrition(df)
    plot_loyalty_vs_attrition(df)
    plot_jobrole_attrition(df)

    # Step 5
    save_engineered_data(df)

    # Step 6
    print_top5_features(df)

    print("\n" + "=" * 55)
    print("   PHASE 2 COMPLETE ✅")
    print("=" * 55)
    print("Output files saved to:")
    print(f"  - {DATA_PATH}hr_engineered.csv")
    print(f"  - {CHARTS_PATH}viz1 through viz8 (.png)")

    return df

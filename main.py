# =============================================================
# main.py
# AI-Powered HR Analytics System — Cloud DB Project 3
# Entry point for running all project phases
# =============================================================

import pandas as pd
from src.preprocessing.preprocessing import run_preprocessing
from src.features.eda import run_eda
from src.models.models import run_models
from src.clustering.kmeans_clustering import run_clustering


def main():

    # ─────────────────────────────────────────
    # PHASE 1 — Data Ingestion & Preprocessing
    # Member 1: Data Engineer
    # ─────────────────────────────────────────
    results = run_preprocessing(
        file_path="Data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv",
        save_path="Data/processed/",
    )

    # ─────────────────────────────────────────
    # PHASE 2 — EDA & Feature Engineering
    # Member 2: Data Analyst
    # ─────────────────────────────────────────
    df = run_eda(
        file_path="Data/processed/hr_cleaned.csv",
    )

    # ─────────────────────────────────────────
    # PHASE 3 — Model Development
    # Member 3: ML Engineer
    # ─────────────────────────────────────────
    run_models(
        data_path="Data/processed/",
        artifact_dir="Data/outputs/",
    )


    # ─────────────────────────────────────────
    # PHASE 4 — Clustering + Bonus
    # Member 4: Cloud & BI Engineer
    # ─────────────────────────────────────────
    print("=======================================================")
    print("   PHASE 4 — CLUSTERING & EMPLOYEE SEGMENTATION")
    print("=======================================================")
    print("\nLoading dataset for clustering...")
    df["EngagementScore"] = (
        df["JobSatisfaction"]
        + df["RelationshipSatisfaction"]
        + df["EnvironmentSatisfaction"]
    ) / 3

    clustered_df, model = run_clustering(df)
    print("Clustering completed successfully\n")

    clustered_df.to_csv("Data/outputs/clustered_employees.csv", index=False)
    print("Clustered file saved successfully")

    features = [
        "MonthlyIncome",
        "JobSatisfaction",
        "PerformanceRating",
        "YearsAtCompany",
        "EngagementScore",
    ]

    cluster_summary = clustered_df.groupby("Cluster")[features].mean()
    print("\nCluster Summary:")
    print(cluster_summary)

    cluster_summary.to_csv("Data/outputs/cluster_summary.csv")
    print("Cluster summary saved successfully")
    print("\n=======================================================")
    print("   ✅ PROJECT COMPLETE — HR ANALYTICS SYSTEM DONE")
    print("=======================================================")


if __name__ == "__main__":
    main()

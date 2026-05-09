# =============================================================
# preprocessing.py
# Phase 1 — Data Ingestion & Preprocessing
# Member 1: Data Engineer
# Project: AI-Powered HR Analytics System
# =============================================================

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# ─────────────────────────────────────────
# Step 1: Load Dataset
# ─────────────────────────────────────────
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the IBM HR Analytics CSV dataset into a Pandas DataFrame.

    Args:
        file_path (str): Path to the raw CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(file_path)
    print(f"[load_data] Dataset loaded — Shape: {df.shape}")
    return df


# ─────────────────────────────────────────
# Step 2: Inspect Data Quality
# ─────────────────────────────────────────
def inspect_data(df: pd.DataFrame) -> None:
    """
    Print missing values, data types, and useless columns
    (columns with only one unique value).

    Args:
        df (pd.DataFrame): Input DataFrame.
    """
    print("\n[inspect_data] === Missing Values ===")
    print(df.isnull().sum())

    print("\n[inspect_data] === Data Types ===")
    print(df.dtypes)

    print("\n[inspect_data] === Useless Columns (1 unique value) ===")
    for col in df.columns:
        if df[col].nunique() == 1:
            print(f"  - {col}: always = {df[col].unique()[0]}")


# ─────────────────────────────────────────
# Step 3: Drop Useless Columns
# ─────────────────────────────────────────
def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that carry no useful information
    (EmployeeCount, Over18, StandardHours all have only 1 unique value).

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with useless columns removed.
    """
    cols_to_drop = ["EmployeeCount", "Over18", "StandardHours"]
    df = df.drop(columns=cols_to_drop)
    print(f"[drop_useless_columns] Dropped {cols_to_drop} — New shape: {df.shape}")
    return df


# ─────────────────────────────────────────
# Step 4: Encode Categorical Variables
# ─────────────────────────────────────────
def encode_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns:
    - Label Encoding for binary columns (Attrition, Gender, OverTime)
    - One-Hot Encoding for multi-value columns

    Encoding map:
        Attrition : No=0, Yes=1
        Gender    : Female=0, Male=1
        OverTime  : No=0, Yes=1

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Encoded DataFrame.
    """
    le = LabelEncoder()

    # Binary columns — Label Encoding
    binary_cols = ["Attrition", "Gender", "OverTime"]
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])
        print(f"[encode_categories] {col} label-encoded: {dict(enumerate(le.classes_))}")

    # Multi-value columns — One-Hot Encoding
    multi_cols = [
        "BusinessTravel",
        "Department",
        "EducationField",
        "JobRole",
        "MaritalStatus",
    ]
    df = pd.get_dummies(df, columns=multi_cols)
    print(f"[encode_categories] One-Hot Encoding applied — New shape: {df.shape}")
    return df


# ─────────────────────────────────────────
# Step 5: Split Features and Target
# ─────────────────────────────────────────
def split_features_target(df: pd.DataFrame, target_col: str = "Attrition"):
    """
    Separate the DataFrame into features (X) and target (y).

    Args:
        df (pd.DataFrame): Encoded DataFrame.
        target_col (str): Name of the target column. Default is 'Attrition'.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    print(f"[split_features_target] Features: {X.shape} | Target distribution:\n{y.value_counts()}")
    return X, y


# ─────────────────────────────────────────
# Step 6: Scale Numerical Features
# ─────────────────────────────────────────
def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Apply StandardScaler to numerical columns.
    Scaler is fit on training data only to prevent data leakage.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (X_train_scaled, X_test_scaled)
    """
    scaler = StandardScaler()
    numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    print(f"[scale_features] Scaled {len(numerical_cols)} numerical columns")
    return X_train, X_test


# ─────────────────────────────────────────
# Step 7: Train/Test Split
# ─────────────────────────────────────────
def split_train_test(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """
    Split data into training and test sets (80/20 by default).
    Stratified split is used to preserve class distribution.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        test_size (float): Fraction of data to use for testing.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"[split_train_test] Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────
# Step 8: Handle Class Imbalance (SMOTE)
# ─────────────────────────────────────────
def apply_smote(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Apply SMOTE (Synthetic Minority Oversampling Technique)
    to balance the training set.

    ⚠️ SMOTE is applied ONLY on training data — never on test data.

    Before SMOTE: Stayed=986 (83.9%), Left=190 (16.1%)
    After SMOTE:  Stayed=986 (50%),  Left=986 (50%)

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X_train_balanced, y_train_balanced)
    """
    print(f"[apply_smote] Before — Stayed: {sum(y_train == 0)} | Left: {sum(y_train == 1)}")
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    print(f"[apply_smote] After  — Stayed: {sum(y_balanced == 0)} | Left: {sum(y_balanced == 1)}")
    print(f"[apply_smote] New balanced train size: {X_balanced.shape}")
    return X_balanced, y_balanced


# ─────────────────────────────────────────
# Step 9: Save Output Files
# ─────────────────────────────────────────
def save_outputs(
    X_train_balanced: pd.DataFrame,
    y_train_balanced: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_cleaned: pd.DataFrame,
    save_path: str = "Data/processed/",
) -> None:
    """
    Save all preprocessed files to disk for the team.

    Files saved:
        - X_train_balanced.csv  → Member 3 (ML Models)
        - y_train_balanced.csv  → Member 3 (ML Models)
        - X_test.csv            → Member 3 (ML Models)
        - y_test.csv            → Member 3 (ML Models)
        - hr_cleaned.csv        → Member 2 (EDA) & Member 4 (Clustering)

    Args:
        save_path (str): Directory path to save files.
    """
    import os
    os.makedirs(save_path, exist_ok=True)

    X_train_balanced.to_csv(save_path + "X_train_balanced.csv", index=False)
    y_train_balanced.to_csv(save_path + "y_train_balanced.csv", index=False)
    X_test.to_csv(save_path + "X_test.csv", index=False)
    y_test.to_csv(save_path + "y_test.csv", index=False)
    df_cleaned.to_csv(save_path + "hr_cleaned.csv", index=False)

    print(f"\n[save_outputs] All files saved to: {save_path}")
    print("  - X_train_balanced.csv  → Member 3")
    print("  - y_train_balanced.csv  → Member 3")
    print("  - X_test.csv            → Member 3")
    print("  - y_test.csv            → Member 3")
    print("  - hr_cleaned.csv        → Member 2 & Member 4")


# ─────────────────────────────────────────
# Main Pipeline Function
# ─────────────────────────────────────────
def run_preprocessing(
    file_path: str,
    save_path: str = "Data/processed/",
) -> dict:
    """
    Run the full Phase 1 preprocessing pipeline end-to-end.

    Steps:
        1. Load data
        2. Inspect data quality
        3. Drop useless columns
        4. Encode categorical variables
        5. Split features and target
        6. Train/test split
        7. Scale numerical features
        8. Apply SMOTE to balance classes
        9. Save output files

    Args:
        file_path (str): Path to raw CSV file.
        save_path (str): Directory to save processed files.

    Returns:
        dict: Dictionary with all processed datasets.
    """
    print("=" * 55)
    print("   PHASE 1 — DATA INGESTION & PREPROCESSING")
    print("=" * 55)

    # Step 1
    df = load_data(file_path)

    # Step 2
    inspect_data(df)

    # Step 3
    df = drop_useless_columns(df)

    # Step 4
    df = encode_categories(df)

    # Step 5
    X, y = split_features_target(df)

    # Step 6
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Step 7
    X_train, X_test = scale_features(X_train, X_test)

    # Step 8
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

    # Step 9 — save hr_cleaned before balancing (full original cleaned data)
    df_cleaned = X.copy()
    df_cleaned["Attrition"] = y
    save_outputs(X_train_balanced, y_train_balanced, X_test, y_test, df_cleaned, save_path)

    print("\n" + "=" * 55)
    print("   PHASE 1 COMPLETE ✅")
    print("=" * 55)

    return {
        "X_train": X_train_balanced,
        "y_train": y_train_balanced,
        "X_test": X_test,
        "y_test": y_test,
        "df_cleaned": df_cleaned,
    }

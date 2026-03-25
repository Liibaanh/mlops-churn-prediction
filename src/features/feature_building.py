"""
Feature engineering for Telco Customer Churn dataset.

Transforms raw CSV data into model-ready features:
  - Fix TotalCharges (string → float)
  - Encode binary columns (Yes/No → 0/1)
  - Encode categorical columns (one-hot encoding)
  - Normalize numeric columns (StandardScaler)
  - Split into train/test sets
  - Save processed data to data/processed/

Usage:
    python -m src.features.build_features
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

RAW_PATH       = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROCESSED_PATH = Path("data/processed")

# ── Column definitions ────────────────────────────────────────────────────────

# Columns to drop — not predictive
DROP_COLS = ["customerID"]

# Target column
TARGET = "Churn"

# Binary Yes/No columns → encode as 0/1
BINARY_COLS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Categorical columns → one-hot encoding
CATEGORICAL_COLS = [
    "gender",
    "MultipleLines",
    "InternetService",
    "Contract",
    "PaymentMethod",
]

# Numeric columns → StandardScaler
NUMERIC_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


def run(test_size: float = 0.2, random_state: int = 42) -> None:
    """
    Full feature engineering pipeline.

    Args:
        test_size:    Fraction of data for test set (default 20%)
        random_state: Random seed for reproducibility
    """
    logger.info("Loading raw data from %s", RAW_PATH)
    df = pd.read_csv(RAW_PATH)
    logger.info("Loaded %d rows, %d columns", df.shape[0], df.shape[1])

    # Run transformations in order
    df = _fix_total_charges(df)
    df = _encode_target(df)
    df = _drop_columns(df)
    df = _encode_binary(df)
    df = _encode_categorical(df)
    df, scaler = _scale_numeric(df)

    # Split into train and test
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # Keep churn ratio equal in train and test
    )

    logger.info("Train set: %d rows", len(X_train))
    logger.info("Test set:  %d rows", len(X_test))
    logger.info("Features:  %d columns", X_train.shape[1])
    logger.info(
        "Churn rate — train: %.1f%%, test: %.1f%%",
        y_train.mean() * 100,
        y_test.mean() * 100,
    )

    # Save processed data
    _save(X_train, X_test, y_train, y_test)


# ── Transformation steps ──────────────────────────────────────────────────────

def _fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    TotalCharges is stored as string in the CSV.
    Some rows have empty strings (new customers with tenure=0).
    Convert to float and fill empty values with 0.
    """
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    empty_count = df["TotalCharges"].isna().sum()
    if empty_count > 0:
        logger.warning(
            "Found %d empty TotalCharges — filling with 0 (new customers)",
            empty_count,
        )
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    return df


def _encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode target variable: Yes → 1, No → 0.
    PyTorch needs numeric labels for binary classification.
    """
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def _drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are not useful for prediction."""
    return df.drop(columns=DROP_COLS)


def _encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode Yes/No columns as 1/0.
    Some columns have 'No internet service' or 'No phone service'
    which we treat as 0 (same as No).
    """
    for col in BINARY_COLS:
        df[col] = df[col].map(
            lambda x: 1 if x == "Yes" else 0
        )
    return df


def _encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical columns.
    drop_first=True avoids multicollinearity
    (e.g. gender_Male is redundant if we have gender_Female).
    """
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    return df


def _scale_numeric(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Normalize numeric columns using StandardScaler.
    This is critical for neural networks — features on different scales
    (tenure: 0-72, TotalCharges: 0-8000) cause slow or unstable training.
    After scaling: mean=0, std=1 for each column.
    """
    scaler = StandardScaler()
    df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])
    return df, scaler


def _save(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Save processed datasets to data/processed/."""
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(PROCESSED_PATH / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_PATH  / "X_test.csv",  index=False)
    y_train.to_csv(PROCESSED_PATH / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_PATH  / "y_test.csv",  index=False)

    logger.info("Saved processed data to %s", PROCESSED_PATH)
    logger.info("Files: X_train.csv, X_test.csv, y_train.csv, y_test.csv")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run()
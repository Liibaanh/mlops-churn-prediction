"""
Data exploration script for Telco Customer Churn dataset.
Run this to understand the data before building the pipeline.

Usage:
    python notebooks/explore_data.py
"""
import pandas as pd

# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ── Basic info ────────────────────────────────────────────────────────────────

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"Rows:    {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nColumn names and types:")
for col in df.columns:
    print(f"  {col}: {df[col].dtype}")

# ── Target variable ───────────────────────────────────────────────────────────

print("\n" + "=" * 50)
print("TARGET: Churn distribution")
print("=" * 50)
print(df["Churn"].value_counts())
print(f"\nChurn rate: {df['Churn'].value_counts(normalize=True)['Yes']:.1%}")

# ── Missing values ────────────────────────────────────────────────────────────

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "No missing values!")

# ── Numeric columns ───────────────────────────────────────────────────────────

print("\n" + "=" * 50)
print("NUMERIC COLUMNS — summary stats")
print("=" * 50)
print(df.describe())

# ── Sample rows ───────────────────────────────────────────────────────────────

print("\n" + "=" * 50)
print("SAMPLE ROWS")
print("=" * 50)
print(df.head(3).to_string())
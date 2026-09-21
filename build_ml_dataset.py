"""
HR Attrition - Build ML Dataset
Prepares a model-ready feature matrix (X) and target vector (y).

Steps:
  1. Load hr_attrition_features.csv
  2. Drop leaky / identifier / derived-duplicate columns
  3. Encode categoricals  — one-hot for nominal, ordinal integer for ordered
  4. Convert target       — Attrition True/False → 1/0
  5. Separate X and y
  6. Report shape, feature list, and class balance
  7. Save X, y, and the combined ml_dataset.csv
"""

import sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

DIVIDER = "=" * 60
SEP     = "-" * 60

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv("hr_attrition_features.csv")
df["Attrition"] = df["Attrition"].astype(str).map({"True": 1, "False": 0})
df["OverTime"]  = df["OverTime"].astype(str).map({"True": 1, "False": 0})

print(DIVIDER)
print(f"Loaded  : {df.shape[0]:,} rows x {df.shape[1]} columns")

# ── 2. Drop columns ───────────────────────────────────────────────────────────
# EmployeeNumber  — identifier, no signal
# SalaryBand, AgeGroup, TenureGroup, PromotionGap — derived from numeric
#   columns already in the dataset; keeping both would double-count the
#   signal and risk data leakage between the bands and their source columns.
#   The richer numeric originals (MonthlyIncome, Age, YearsAtCompany,
#   YearsSinceLastPromotion) are retained instead.
DROP_COLS = [
    "EmployeeNumber",
    "SalaryBand",
    "AgeGroup",
    "TenureGroup",
    "PromotionGap",
]
df = df.drop(columns=DROP_COLS)
print(f"Dropped : {DROP_COLS}")

# ── 3. Encode categoricals ────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("ENCODING")
print(SEP)

# 3a. Ordinal columns — map to their natural integer scale (already numeric
#     in the CSV; just confirm dtype and print for transparency)
ordinal_cols = [
    "Education",               # 1–5
    "EnvironmentSatisfaction", # 1–4
    "JobInvolvement",          # 1–4
    "JobLevel",                # 1–5
    "JobSatisfaction",         # 1–4
    "PerformanceRating",       # 1–4
    "RelationshipSatisfaction",# 1–4
    "StockOptionLevel",        # 0–3
    "WorkLifeBalance",         # 1–4
]
for col in ordinal_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype(int)
    print(f"  {col:<30} kept as int  (ordinal scale)")

# 3b. Binary string columns — already converted above
print(f"  {'OverTime':<30} kept as int  (0/1 binary)")

# 3c. Nominal string columns — one-hot encode, drop_first to avoid
#     multicollinearity (dummy variable trap)
nominal_cols = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
]
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True, dtype=int)
new_dummies = [c for c in df.columns if any(c.startswith(n + "_") for n in nominal_cols)]
print(f"\n  One-hot encoded {len(nominal_cols)} nominal columns → {len(new_dummies)} dummy features:")
for col in new_dummies:
    print(f"    {col}")

# ── 4. Separate X and y ───────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("X / y SPLIT")
print(SEP)

y = df["Attrition"].astype(int)
X = df.drop(columns=["Attrition"])

print(f"  y (target)  : 'Attrition'  shape={y.shape}")
print(f"  X (features): shape={X.shape}")

# ── 5. Feature inventory ──────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("FEATURE INVENTORY")
print(SEP)
print(f"  Total features : {X.shape[1]}")
print()
for i, col in enumerate(X.columns, 1):
    dtype = X[col].dtype
    print(f"  {i:>3}. {col:<40} {dtype}")

# ── 6. Class balance ──────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("CLASS BALANCE")
print(SEP)
counts = y.value_counts().sort_index()
pcts   = y.value_counts(normalize=True).sort_index().mul(100).round(2)
balance = pd.DataFrame({"Label": ["Stayed (0)", "Left (1)"],
                         "Count": counts.values,
                         "Pct %": pcts.values})
print(balance.to_string(index=False))
ratio = counts[0] / counts[1]
print(f"\n  Imbalance ratio (majority:minority) = {ratio:.1f}:1")
print(f"  Note: stratified train/test split and class_weight='balanced'")
print(f"        or SMOTE recommended at modelling stage.")

# ── 7. Verify no missing values remain ───────────────────────────────────────
print(f"\n{DIVIDER}")
print("MISSING VALUE CHECK")
print(SEP)
total_missing = X.isna().sum().sum()
if total_missing == 0:
    print("  X : 0 missing values — ready for modelling.")
else:
    cols_with_na = X.isna().sum()[X.isna().sum() > 0]
    print(f"  WARNING: {total_missing} missing values found:")
    print(cols_with_na.to_string())

# ── 8. Save ───────────────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("SAVE")
print(SEP)

ml_df = X.copy()
ml_df.insert(0, "Attrition", y)
ml_df.to_csv("ml_dataset.csv", index=False)
print(f"  ml_dataset.csv  saved  ({ml_df.shape[0]:,} rows x {ml_df.shape[1]} columns)")

print(f"\n{DIVIDER}")
print("BUILD COMPLETE")
print(f"  X shape : {X.shape}")
print(f"  y shape : {y.shape}")
print(DIVIDER)

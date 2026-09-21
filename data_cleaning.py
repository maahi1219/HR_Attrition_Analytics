"""
HR Attrition Dataset - Data Cleaning
Steps:
  1. Load raw data
  2. Strip BOM / whitespace from column names
  3. Missing values  - check & strategy
  4. Duplicate rows  - check & drop
  5. Invalid values  - numeric bounds + invalid categoricals
  6. Data-type conversion
  7. Drop zero-variance columns
  8. Save cleaned dataset
"""

import sys
import pandas as pd
import numpy as np

# Force UTF-8 output so box-drawing and arrow characters print correctly
# on Windows terminals that default to cp1252.
sys.stdout.reconfigure(encoding="utf-8")

DIVIDER = "=" * 60
SEP     = "-" * 60

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
print(DIVIDER)
print(f"Loaded  : {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── 2. Strip BOM / leading-trailing whitespace from column names ──────────────
df.columns = df.columns.str.strip().str.lstrip("\ufeff")
print(f"Columns after name-clean: {df.columns.tolist()}")

# ═══════════════════════════════════════════════════════
# STEP 3 — MISSING VALUES
# ═══════════════════════════════════════════════════════
print(DIVIDER)
print("STEP 3 — MISSING VALUES")
print(SEP)

missing = df.isna().sum()
missing = missing[missing > 0]

if missing.empty:
    print("  No missing values — nothing to impute or drop.")
else:
    print(missing.to_string())
    # Strategy per column would go here, e.g.:
    #   Numeric  → median imputation  (robust to skew)
    #   Category → mode imputation    (most-frequent label)
    for col in missing.index:
        if df[col].dtype == "object":
            fill_val = df[col].mode()[0]
            strategy = f"mode imputed → '{fill_val}'"
        else:
            fill_val = df[col].median()
            strategy = f"median imputed → {fill_val}"
        df[col] = df[col].fillna(fill_val)
        print(f"  {col:<30} {strategy}")

print(f"  Missing after cleaning : {df.isna().sum().sum()}")

# ═══════════════════════════════════════════════════════
# STEP 4 — DUPLICATE ROWS
# ═══════════════════════════════════════════════════════
print(DIVIDER)
print("STEP 4 — DUPLICATE ROWS")
print(SEP)

n_before = len(df)
df = df.drop_duplicates()
n_dropped = n_before - len(df)
print(f"  Duplicates removed : {n_dropped:,}")
print(f"  Rows remaining     : {len(df):,}")

# ═══════════════════════════════════════════════════════
# STEP 5 — INVALID VALUES
# ═══════════════════════════════════════════════════════
print(DIVIDER)
print("STEP 5 — INVALID VALUES")
print(SEP)

issues_found = 0

# ── 5a. Age — valid human working range 18–65 ─────────────────────────────────
mask = ~df["Age"].between(18, 65)
if mask.any():
    print(f"  Age out of range [18–65] : {mask.sum()} rows — DROPPED")
    df = df[~mask]
    issues_found += mask.sum()
else:
    print("  Age [18–65]              : OK")

# ── 5b. MonthlyIncome — must be positive ─────────────────────────────────────
mask = df["MonthlyIncome"] <= 0
if mask.any():
    print(f"  MonthlyIncome ≤ 0        : {mask.sum()} rows — DROPPED")
    df = df[~mask]
    issues_found += mask.sum()
else:
    print("  MonthlyIncome > 0        : OK")

# ── 5c. DailyRate / HourlyRate / MonthlyRate — must be positive ──────────────
for rate_col in ["DailyRate", "HourlyRate", "MonthlyRate"]:
    mask = df[rate_col] <= 0
    if mask.any():
        print(f"  {rate_col} ≤ 0 : {mask.sum()} rows — DROPPED")
        df = df[~mask]
        issues_found += mask.sum()
    else:
        print(f"  {rate_col} > 0{' ' * (22 - len(rate_col))}: OK")

# ── 5d. Tenure logic: YearsAtCompany ≤ TotalWorkingYears ─────────────────────
mask = df["YearsAtCompany"] > df["TotalWorkingYears"]
if mask.any():
    print(f"  YearsAtCompany > TotalWorkingYears : {mask.sum()} rows — DROPPED")
    df = df[~mask]
    issues_found += mask.sum()
else:
    print("  Tenure logic (YearsAtCompany <= TotalWorkingYears) : OK")

# ── 5e. YearsInCurrentRole ≤ YearsAtCompany ──────────────────────────────────
mask = df["YearsInCurrentRole"] > df["YearsAtCompany"]
if mask.any():
    print(f"  YearsInCurrentRole > YearsAtCompany : {mask.sum()} rows — DROPPED")
    df = df[~mask]
    issues_found += mask.sum()
else:
    print("  YearsInCurrentRole <= YearsAtCompany              : OK")

# ── 5f. YearsWithCurrManager ≤ YearsAtCompany ────────────────────────────────
mask = df["YearsWithCurrManager"] > df["YearsAtCompany"]
if mask.any():
    print(f"  YearsWithCurrManager > YearsAtCompany : {mask.sum()} rows — DROPPED")
    df = df[~mask]
    issues_found += mask.sum()
else:
    print("  YearsWithCurrManager <= YearsAtCompany            : OK")

# ── 5g. Categorical columns — valid-set validation ────────────────────────────
valid_categories = {
    "Attrition"      : {"Yes", "No"},
    "BusinessTravel" : {"Non-Travel", "Travel_Rarely", "Travel_Frequently"},
    "Department"     : {"Sales", "Research & Development", "Human Resources"},
    "Gender"         : {"Male", "Female"},
    "MaritalStatus"  : {"Single", "Married", "Divorced"},
    "OverTime"       : {"Yes", "No"},
    "Over18"         : {"Y"},
    "EducationField" : {
        "Life Sciences", "Medical", "Marketing",
        "Technical Degree", "Human Resources", "Other"
    },
    "JobRole" : {
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources"
    },
}

for col, valid_set in valid_categories.items():
    invalid_mask = ~df[col].isin(valid_set)
    n_invalid = invalid_mask.sum()
    if n_invalid > 0:
        bad_vals = df.loc[invalid_mask, col].unique().tolist()
        print(f"  {col:<20} invalid values {bad_vals} : {n_invalid} rows — DROPPED")
        df = df[~invalid_mask]
        issues_found += n_invalid
    else:
        print(f"  {col:<20} categories     : OK")

print(f"\n  Total invalid rows removed : {issues_found:,}")

# ── 5h. Ordinal range checks (Likert-style 1–4 or 1–5 scales) ────────────────
ordinal_ranges = {
    "Education"               : (1, 5),
    "EnvironmentSatisfaction" : (1, 4),
    "JobInvolvement"          : (1, 4),
    "JobLevel"                : (1, 5),
    "JobSatisfaction"         : (1, 4),
    "PerformanceRating"       : (1, 4),
    "RelationshipSatisfaction": (1, 4),
    "StockOptionLevel"        : (0, 3),
    "WorkLifeBalance"         : (1, 4),
}
for col, (lo, hi) in ordinal_ranges.items():
    mask = ~df[col].between(lo, hi)
    if mask.any():
        print(f"  {col} out of [{lo}–{hi}] : {mask.sum()} rows — DROPPED")
        df = df[~mask]
    else:
        print(f"  {col:<30} [{lo}–{hi}] : OK")

# ═══════════════════════════════════════════════════════
# STEP 6 — DATA-TYPE CONVERSION
# ═══════════════════════════════════════════════════════
print(DIVIDER)
print("STEP 6 — DATA-TYPE CONVERSION")
print(SEP)

# Binary string → bool
binary_map = {"Yes": True, "No": False}
df["Attrition"] = df["Attrition"].map(binary_map)
df["OverTime"]  = df["OverTime"].map(binary_map)
print("  Attrition  → bool (True/False)")
print("  OverTime   → bool (True/False)")

# Nominal strings → pandas Categorical
nominal_cols = [
    "BusinessTravel", "Department", "EducationField",
    "Gender", "JobRole", "MaritalStatus",
]
for col in nominal_cols:
    df[col] = df[col].astype("category")
    print(f"  {col:<25} → category")

# Ordinal ints → pandas ordered Categorical
ordinal_orders = {
    "Education"               : [1, 2, 3, 4, 5],
    "EnvironmentSatisfaction" : [1, 2, 3, 4],
    "JobInvolvement"          : [1, 2, 3, 4],
    "JobLevel"                : [1, 2, 3, 4, 5],
    "JobSatisfaction"         : [1, 2, 3, 4],
    "PerformanceRating"       : [1, 2, 3, 4],
    "RelationshipSatisfaction": [1, 2, 3, 4],
    "StockOptionLevel"        : [0, 1, 2, 3],
    "WorkLifeBalance"         : [1, 2, 3, 4],
}
for col, order in ordinal_orders.items():
    df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    print(f"  {col:<30} → ordered category {order}")

# ═══════════════════════════════════════════════════════
# STEP 7 — DROP ZERO-VARIANCE COLUMNS
# ═══════════════════════════════════════════════════════
print(DIVIDER)
print("STEP 7 — DROP ZERO-VARIANCE COLUMNS")
print(SEP)

zero_var = ["EmployeeCount", "Over18", "StandardHours"]
df = df.drop(columns=zero_var)
print(f"  Dropped : {zero_var}")

# ═══════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════
print(DIVIDER)
print("CLEANING SUMMARY")
print(SEP)
print(f"  Final shape     : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Missing values  : {df.isna().sum().sum()}")
print(f"  Duplicates      : {df.duplicated().sum()}")
print("\n  Final dtypes:")
print(df.dtypes.to_string())

# ═══════════════════════════════════════════════════════
# STEP 8 — SAVE
# ═══════════════════════════════════════════════════════
print(DIVIDER)
print("STEP 8 — SAVE")
print(SEP)
df.to_csv("hr_attrition_clean.csv", index=False)
print("  Saved → hr_attrition_clean.csv")
print(DIVIDER)

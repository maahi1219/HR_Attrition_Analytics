"""
HR Attrition Dataset — Data Audit
Automatically reports shape, types, missing values, duplicates,
unique categories, min/max values, and class distribution.
"""

import pandas as pd

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

DIVIDER = "=" * 60
SEP     = "-" * 60

# ── 1. Shape ──────────────────────────────────────────────────────────────────
print(DIVIDER)
print("1. SHAPE")
print(SEP)
rows, cols = df.shape
print(f"  Rows    : {rows:,}")
print(f"  Columns : {cols}")

# ── 2. Column names ───────────────────────────────────────────────────────────
print(DIVIDER)
print("2. COLUMN NAMES")
print(SEP)
for i, col in enumerate(df.columns, 1):
    print(f"  {i:>2}. {col}")

# ── 3. Data types ─────────────────────────────────────────────────────────────
print(DIVIDER)
print("3. DATA TYPES")
print(SEP)
print(df.dtypes.to_string())

# ── 4. Missing values ─────────────────────────────────────────────────────────
print(DIVIDER)
print("4. MISSING VALUES")
print(SEP)
missing     = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df  = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
missing_df  = missing_df[missing_df["Missing Count"] > 0]

if missing_df.empty:
    print("  No missing values found.")
else:
    print(missing_df.to_string())

print(f"\n  Total cells   : {df.size:,}")
print(f"  Total missing : {df.isna().sum().sum():,}")

# ── 5. Duplicate rows ─────────────────────────────────────────────────────────
print(DIVIDER)
print("5. DUPLICATE ROWS")
print(SEP)
n_dupes = df.duplicated().sum()
print(f"  Duplicate rows : {n_dupes:,}")

# ── 6. Unique categories (string / low-cardinality columns) ───────────────────
print(DIVIDER)
print("6. UNIQUE CATEGORIES  (string & low-cardinality numeric columns)")
print(SEP)
cat_cols = df.select_dtypes(include=["str", "object", "category"]).columns.tolist()

# Also surface numeric columns with ≤ 10 distinct values as quasi-categorical
num_quasi = [
    c for c in df.select_dtypes(include="number").columns
    if df[c].nunique() <= 10
]

for col in cat_cols + num_quasi:
    n_unique = df[col].nunique(dropna=False)
    values   = df[col].dropna().unique()
    display  = sorted(str(v) for v in values)
    if len(display) <= 12:
        vals_str = ", ".join(display)
    else:
        vals_str = ", ".join(display[:12]) + f"  … (+{len(display) - 12} more)"
    print(f"  {col:<30} unique={n_unique:>4}   [{vals_str}]")

# ── 7. Numeric summary (min / max / mean / std) ───────────────────────────────
print(DIVIDER)
print("7. NUMERIC SUMMARY  (min / max / mean / std)")
print(SEP)
num_cols = df.select_dtypes(include="number").columns.tolist()
summary  = df[num_cols].agg(["min", "max", "mean", "std"]).T.round(2)
summary.columns = ["Min", "Max", "Mean", "Std"]
print(summary.to_string())

# ── 8. Class distribution — Attrition ────────────────────────────────────────
print(DIVIDER)
print("8. CLASS DISTRIBUTION  — Attrition (target variable)")
print(SEP)
counts = df["Attrition"].value_counts(dropna=False)
pcts   = df["Attrition"].value_counts(normalize=True, dropna=False).mul(100).round(2)
dist   = pd.DataFrame({"Count": counts, "Percent %": pcts})
print(dist.to_string())

# ── 9. Full df.info() ─────────────────────────────────────────────────────────
print(DIVIDER)
print("9. df.info()")
print(SEP)
df.info()

print(DIVIDER)
print("AUDIT COMPLETE")
print(DIVIDER)

"""
HR Attrition - Feature Engineering
Adds meaningful derived features to the cleaned dataset:
  1. SalaryBand        - Low / Medium / High
  2. AgeGroup          - 18-25 / 26-35 / 36-45 / 46-55 / 56+
  3. TenureGroup       - 0-1 / 2-3 / 4-5 / 6-10 / 10+
  4. PromotionGap      - Recent (0-2 yrs) / Long (3+ yrs)

Saves result to: hr_attrition_features.csv
"""

import sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

DIVIDER = "=" * 60
SEP     = "-" * 60

def attrition_rate(series):
    """Return attrition % for a boolean Series."""
    return round(series.mean() * 100, 1)

def profile(df, col):
    """Print count + attrition rate per category, sorted by attrition %."""
    grp = (
        df.groupby(col, observed=True)["Attrition"]
        .agg(Total="count", Left="sum")
        .assign(**{"Attrition %": lambda x: (x["Left"] / x["Total"] * 100).round(1)})
        .sort_values("Attrition %", ascending=False)
    )
    print(grp.to_string())

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("hr_attrition_clean.csv")
df["Attrition"] = df["Attrition"].astype(str).map({"True": True, "False": False})

print(DIVIDER)
print(f"  Loaded : {df.shape[0]:,} rows × {df.shape[1]} columns")

# ═══════════════════════════════════════════════════════
# 1. SALARY BAND
# ═══════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("  1. SALARY BAND")
print(SEP)

# Tertile-based cut so each band has a roughly equal number of employees
tertiles = df["MonthlyIncome"].quantile([1/3, 2/3])
low_cap  = tertiles[1/3]
mid_cap  = tertiles[2/3]

print(f"  MonthlyIncome tertile thresholds:")
print(f"    Low    : <= ${low_cap:,.0f}")
print(f"    Medium : ${low_cap:,.0f} – ${mid_cap:,.0f}")
print(f"    High   : >  ${mid_cap:,.0f}")

df["SalaryBand"] = pd.cut(
    df["MonthlyIncome"],
    bins=[-np.inf, low_cap, mid_cap, np.inf],
    labels=["Low", "Medium", "High"],
)

print(f"\n  Distribution:")
print(df["SalaryBand"].value_counts().sort_index().to_string())
print(f"\n  Attrition by Salary Band:")
profile(df, "SalaryBand")

# ═══════════════════════════════════════════════════════
# 2. AGE GROUP
# ═══════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("  2. AGE GROUP")
print(SEP)

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[17, 25, 35, 45, 55, 65],
    labels=["18-25", "26-35", "36-45", "46-55", "56+"],
)

print(f"  Distribution:")
print(df["AgeGroup"].value_counts().sort_index().to_string())
print(f"\n  Attrition by Age Group:")
profile(df, "AgeGroup")

# ═══════════════════════════════════════════════════════
# 3. TENURE GROUP
# ═══════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("  3. TENURE GROUP  (YearsAtCompany)")
print(SEP)

df["TenureGroup"] = pd.cut(
    df["YearsAtCompany"],
    bins=[-1, 1, 3, 5, 10, np.inf],
    labels=["0-1", "2-3", "4-5", "6-10", "10+"],
)

print(f"  Distribution:")
print(df["TenureGroup"].value_counts().sort_index().to_string())
print(f"\n  Attrition by Tenure Group:")
profile(df, "TenureGroup")

# ═══════════════════════════════════════════════════════
# 4. PROMOTION GAP
# ═══════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("  4. PROMOTION GAP  (YearsSinceLastPromotion)")
print(SEP)

df["PromotionGap"] = pd.cut(
    df["YearsSinceLastPromotion"],
    bins=[-1, 2, np.inf],
    labels=["Recent (0-2 yrs)", "Long (3+ yrs)"],
)

print(f"  Distribution:")
print(df["PromotionGap"].value_counts().sort_index().to_string())
print(f"\n  Attrition by Promotion Gap:")
profile(df, "PromotionGap")

# ═══════════════════════════════════════════════════════
# SUMMARY — all new features
# ═══════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("  SUMMARY — NEW FEATURES ADDED")
print(SEP)
new_cols = ["SalaryBand", "AgeGroup", "TenureGroup", "PromotionGap"]
for col in new_cols:
    dtype  = df[col].dtype
    cats   = list(df[col].cat.categories)
    nulls  = df[col].isna().sum()
    print(f"  {col:<16} dtype={dtype}  categories={cats}  nulls={nulls}")

print(f"\n  Final shape : {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv("hr_attrition_features.csv", index=False)
print(f"\n  Saved → hr_attrition_features.csv")
print(DIVIDER)

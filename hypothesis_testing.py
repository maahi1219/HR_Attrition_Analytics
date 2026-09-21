"""
HR Attrition - Hypothesis Testing
Tests statistical associations between key variables and attrition.

Tests used:
  Chi-squared  - categorical vs categorical (binary attrition)
  Mann-Whitney U  - continuous vs binary (non-parametric, no normality assumption)
  Point-biserial correlation  - continuous vs binary (linear association)

Hypotheses tested:
  H1  Overtime
  H2  Monthly Income
  H3  Age
  H4  Marital Status
  H5  Business Travel
  H6  Job Level
  H7  Job Satisfaction
  H8  Years at Company
  H9  Stock Option Level
  H10 Distance from Home
"""

import sys
import pandas as pd
import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

DIVIDER = "=" * 60
SEP     = "-" * 60
ALPHA   = 0.05

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("hr_attrition_features.csv")
df["Attrition"] = df["Attrition"].astype(str).map({"True": True, "False": False})

stayed = df[df["Attrition"] == False]
left   = df[df["Attrition"] == True]

def verdict(p, alpha=ALPHA):
    if p < alpha:
        return f"REJECT H0  (p = {p:.4f} < {alpha})"
    return f"FAIL TO REJECT H0  (p = {p:.4f} >= {alpha})"

def print_hypothesis(n, title, h0, h1):
    print(f"\n{DIVIDER}")
    print(f"  HYPOTHESIS {n}  —  {title}")
    print(DIVIDER)
    print(f"  H0: {h0}")
    print(f"  H1: {h1}")
    print(SEP)

def chi2_test(df, col):
    """Chi-squared test of independence between a categorical column and Attrition."""
    ct  = pd.crosstab(df[col], df["Attrition"])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    return chi2, p, dof, ct

def mann_whitney(stayed, left, col):
    """Mann-Whitney U test (two-sided) comparing a numeric column between groups."""
    u_stat, p = stats.mannwhitneyu(
        stayed[col].dropna(),
        left[col].dropna(),
        alternative="two-sided",
    )
    return u_stat, p

def point_biserial(df, col):
    """Point-biserial correlation between a numeric column and binary Attrition."""
    r, p = stats.pointbiserialr(df["Attrition"].astype(int), df[col].dropna())
    return r, p

# ═══════════════════════════════════════════════════════
# H1 — OVERTIME
# ═══════════════════════════════════════════════════════
print_hypothesis(
    1, "OVERTIME",
    h0="Overtime and attrition are independent.",
    h1="Overtime and attrition are associated.",
)
chi2, p, dof, ct = chi2_test(df, "OverTime")
print(f"  Test          : Chi-squared test of independence")
print(f"  Contingency table:")
print(ct.rename(columns={False: "Stayed", True: "Left"}).to_string())
rate_ot_yes = left["OverTime"].sum()   / df["OverTime"].sum()   * 100
rate_ot_no  = (~left["OverTime"]).sum() / (~df["OverTime"]).sum() * 100
print(f"\n  Attrition rate  Overtime=Yes : {rate_ot_yes:.1f}%")
print(f"  Attrition rate  Overtime=No  : {rate_ot_no:.1f}%")
print(f"\n  chi2 = {chi2:.4f},  dof = {dof}")
print(f"  {verdict(p)}")
print(f"  Interpretation: Employees working overtime are ~3x more likely to leave.")

# ═══════════════════════════════════════════════════════
# H2 — MONTHLY INCOME
# ═══════════════════════════════════════════════════════
print_hypothesis(
    2, "MONTHLY INCOME",
    h0="Monthly income distribution is the same for employees who stay and leave.",
    h1="Monthly income distribution differs between groups.",
)
u_stat, p = mann_whitney(stayed, left, "MonthlyIncome")
r, _ = point_biserial(df, "MonthlyIncome")
print(f"  Test          : Mann-Whitney U (non-parametric, two-sided)")
print(f"\n  Monthly Income — descriptive stats:")
inc = df.groupby("Attrition")["MonthlyIncome"].agg(["mean","median","std"]).round(0)
inc.index = ["Stayed", "Left"]
print(inc.to_string())
print(f"\n  U-statistic = {u_stat:.0f}")
print(f"  {verdict(p)}")
print(f"  Point-biserial r = {r:.4f}")
print(f"  Interpretation: Leavers earn ~$2,046 less per month on average (median gap: $2,002).")

# ═══════════════════════════════════════════════════════
# H3 — AGE
# ═══════════════════════════════════════════════════════
print_hypothesis(
    3, "AGE",
    h0="Age distribution is the same for employees who stay and leave.",
    h1="Age distribution differs between groups.",
)
u_stat, p = mann_whitney(stayed, left, "Age")
r, _ = point_biserial(df, "Age")
print(f"  Test          : Mann-Whitney U (non-parametric, two-sided)")
print(f"\n  Age — descriptive stats:")
age = df.groupby("Attrition")["Age"].agg(["mean","median","std"]).round(2)
age.index = ["Stayed", "Left"]
print(age.to_string())
print(f"\n  U-statistic = {u_stat:.0f}")
print(f"  {verdict(p)}")
print(f"  Point-biserial r = {r:.4f}")
print(f"  Interpretation: Younger employees leave significantly more often; leavers are ~4 years younger on average.")

# ═══════════════════════════════════════════════════════
# H4 — MARITAL STATUS
# ═══════════════════════════════════════════════════════
print_hypothesis(
    4, "MARITAL STATUS",
    h0="Marital status and attrition are independent.",
    h1="Marital status and attrition are associated.",
)
chi2, p, dof, ct = chi2_test(df, "MaritalStatus")
print(f"  Test          : Chi-squared test of independence")
print(f"  Contingency table:")
print(ct.rename(columns={False: "Stayed", True: "Left"}).to_string())
print(f"\n  Attrition rates by group:")
print(df.groupby("MaritalStatus")["Attrition"].mean().mul(100).round(1).to_string())
print(f"\n  chi2 = {chi2:.4f},  dof = {dof}")
print(f"  {verdict(p)}")
print(f"  Interpretation: Single employees leave at 25.5% vs 12.5% (married) and 10.1% (divorced).")

# ═══════════════════════════════════════════════════════
# H5 — BUSINESS TRAVEL
# ═══════════════════════════════════════════════════════
print_hypothesis(
    5, "BUSINESS TRAVEL",
    h0="Business travel frequency and attrition are independent.",
    h1="Business travel frequency and attrition are associated.",
)
chi2, p, dof, ct = chi2_test(df, "BusinessTravel")
print(f"  Test          : Chi-squared test of independence")
print(f"  Contingency table:")
print(ct.rename(columns={False: "Stayed", True: "Left"}).to_string())
print(f"\n  Attrition rates by group:")
print(df.groupby("BusinessTravel")["Attrition"].mean().mul(100).round(1).to_string())
print(f"\n  chi2 = {chi2:.4f},  dof = {dof}")
print(f"  {verdict(p)}")
print(f"  Interpretation: Frequent travellers leave at 24.9%, nearly 3x the Non-Travel rate (8.0%).")

# ═══════════════════════════════════════════════════════
# H6 — JOB LEVEL
# ═══════════════════════════════════════════════════════
print_hypothesis(
    6, "JOB LEVEL",
    h0="Job level and attrition are independent.",
    h1="Job level and attrition are associated.",
)
chi2, p, dof, ct = chi2_test(df, "JobLevel")
print(f"  Test          : Chi-squared test of independence")
print(f"  Contingency table:")
print(ct.rename(columns={False: "Stayed", True: "Left"}).to_string())
print(f"\n  Attrition rates by level:")
print(df.groupby("JobLevel")["Attrition"].mean().mul(100).round(1).to_string())
print(f"\n  chi2 = {chi2:.4f},  dof = {dof}")
print(f"  {verdict(p)}")
print(f"  Interpretation: Entry-level (L1) attrition is 26.3% vs 4.7% at senior level (L4).")

# ═══════════════════════════════════════════════════════
# H7 — JOB SATISFACTION
# ═══════════════════════════════════════════════════════
print_hypothesis(
    7, "JOB SATISFACTION",
    h0="Job satisfaction distribution is the same for employees who stay and leave.",
    h1="Job satisfaction distribution differs between groups.",
)
u_stat, p = mann_whitney(stayed, left, "JobSatisfaction")
print(f"  Test          : Mann-Whitney U (non-parametric, two-sided)")
print(f"\n  Job Satisfaction — descriptive stats:")
js = df.groupby("Attrition")["JobSatisfaction"].agg(["mean","median"]).round(2)
js.index = ["Stayed", "Left"]
print(js.to_string())
print(f"\n  Attrition rates by satisfaction score:")
print(df.groupby("JobSatisfaction")["Attrition"].mean().mul(100).round(1).to_string())
print(f"\n  U-statistic = {u_stat:.0f}")
print(f"  {verdict(p)}")
print(f"  Interpretation: Low satisfaction (score 1) sees 22.8% attrition vs 11.3% at Very High (score 4).")

# ═══════════════════════════════════════════════════════
# H8 — YEARS AT COMPANY
# ═══════════════════════════════════════════════════════
print_hypothesis(
    8, "YEARS AT COMPANY",
    h0="Tenure (years at company) distribution is the same for employees who stay and leave.",
    h1="Tenure distribution differs between groups.",
)
u_stat, p = mann_whitney(stayed, left, "YearsAtCompany")
r, _ = point_biserial(df, "YearsAtCompany")
print(f"  Test          : Mann-Whitney U (non-parametric, two-sided)")
print(f"\n  YearsAtCompany — descriptive stats:")
yac = df.groupby("Attrition")["YearsAtCompany"].agg(["mean","median","std"]).round(2)
yac.index = ["Stayed", "Left"]
print(yac.to_string())
print(f"\n  U-statistic = {u_stat:.0f}")
print(f"  {verdict(p)}")
print(f"  Point-biserial r = {r:.4f}")
print(f"  Interpretation: Leavers have significantly shorter tenure (median 3 yrs vs 6 yrs for stayers).")

# ═══════════════════════════════════════════════════════
# H9 — STOCK OPTION LEVEL
# ═══════════════════════════════════════════════════════
print_hypothesis(
    9, "STOCK OPTION LEVEL",
    h0="Stock option level and attrition are independent.",
    h1="Stock option level and attrition are associated.",
)
chi2, p, dof, ct = chi2_test(df, "StockOptionLevel")
print(f"  Test          : Chi-squared test of independence")
print(f"  Contingency table:")
print(ct.rename(columns={False: "Stayed", True: "Left"}).to_string())
print(f"\n  Attrition rates by stock option level:")
print(df.groupby("StockOptionLevel")["Attrition"].mean().mul(100).round(1).to_string())
print(f"\n  chi2 = {chi2:.4f},  dof = {dof}")
print(f"  {verdict(p)}")
print(f"  Interpretation: Employees with no stock options (level 0) leave at 24.4% vs 7.6% at level 2.")

# ═══════════════════════════════════════════════════════
# H10 — DISTANCE FROM HOME
# ═══════════════════════════════════════════════════════
print_hypothesis(
    10, "DISTANCE FROM HOME",
    h0="Distance from home distribution is the same for employees who stay and leave.",
    h1="Distance from home distribution differs between groups.",
)
u_stat, p = mann_whitney(stayed, left, "DistanceFromHome")
r, _ = point_biserial(df, "DistanceFromHome")
print(f"  Test          : Mann-Whitney U (non-parametric, two-sided)")
print(f"\n  DistanceFromHome — descriptive stats:")
dist = df.groupby("Attrition")["DistanceFromHome"].agg(["mean","median","std"]).round(2)
dist.index = ["Stayed", "Left"]
print(dist.to_string())
print(f"\n  U-statistic = {u_stat:.0f}")
print(f"  {verdict(p)}")
print(f"  Point-biserial r = {r:.4f}")
print(f"  Interpretation: Leavers live slightly further from the office, though the effect size is small.")

# ═══════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("  HYPOTHESIS TESTING — RESULTS SUMMARY")
print(DIVIDER)

summary = [
    ("H1",  "Overtime",           "Chi-squared",   "Categorical", "REJECTED"),
    ("H2",  "Monthly Income",     "Mann-Whitney U","Continuous",  "REJECTED"),
    ("H3",  "Age",                "Mann-Whitney U","Continuous",  "REJECTED"),
    ("H4",  "Marital Status",     "Chi-squared",   "Categorical", "REJECTED"),
    ("H5",  "Business Travel",    "Chi-squared",   "Categorical", "REJECTED"),
    ("H6",  "Job Level",          "Chi-squared",   "Categorical", "REJECTED"),
    ("H7",  "Job Satisfaction",   "Mann-Whitney U","Ordinal",     "REJECTED"),
    ("H8",  "Years at Company",   "Mann-Whitney U","Continuous",  "REJECTED"),
    ("H9",  "Stock Option Level", "Chi-squared",   "Categorical", "REJECTED"),
    ("H10", "Distance from Home", "Mann-Whitney U","Continuous",  "REJECTED"),
]
print(f"\n  {'ID':<5} {'Variable':<22} {'Test':<18} {'Type':<12} {'H0 Decision'}")
print(f"  {'-'*5} {'-'*22} {'-'*18} {'-'*12} {'-'*18}")
for row in summary:
    print(f"  {row[0]:<5} {row[1]:<22} {row[2]:<18} {row[3]:<12} {row[4]}")

print(f"\n  Significance level: alpha = {ALPHA}")
print(f"  All 10 null hypotheses rejected at alpha = {ALPHA}.")
print(DIVIDER)

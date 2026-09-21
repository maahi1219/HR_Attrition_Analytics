"""
HR Attrition - Exploratory Data Analysis
Covers:
  0. Overall attrition rate
  1. Demographics  : Age, Gender, Marital Status, Education
  2. Job            : Department, Job Role, Job Level, Business Travel
  3. Compensation   : Monthly Income, Salary Band, % Salary Hike, Stock Options
  4. Workplace      : Overtime, Job Satisfaction, Environment Satisfaction, Work-Life Balance
  5. Career         : Years at Company, Years in Role, Since Promotion, With Manager
"""

import sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("hr_attrition_clean.csv")
df["Attrition"] = df["Attrition"].map({"True": True, "False": False}).fillna(df["Attrition"])
df["Attrition"] = df["Attrition"].astype(bool)

DIVIDER = "=" * 60
SEP     = "-" * 60

def attrition_table(df, col, label=None):
    """Return a summary DataFrame: count per group, attrition count, attrition rate %."""
    label = label or col
    grp   = df.groupby(col, observed=True)
    total = grp["Attrition"].count().rename("Total")
    left  = grp["Attrition"].sum().rename("Left")
    rate  = (left / total * 100).round(1).rename("Attrition %")
    out   = pd.concat([total, left, rate], axis=1).sort_values("Attrition %", ascending=False)
    return out

def print_section(title):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

# ═══════════════════════════════════════════════════════
# 0. OVERALL ATTRITION RATE
# ═══════════════════════════════════════════════════════
print_section("0. OVERALL ATTRITION RATE")

total_employees = len(df)
employees_left  = df["Attrition"].sum()
attrition_rate  = employees_left / total_employees * 100

print(f"  Total employees  : {total_employees:,}")
print(f"  Employees left   : {int(employees_left):,}")
print(f"  Employees stayed : {total_employees - int(employees_left):,}")
print(f"\n  Attrition Rate = {int(employees_left)} / {total_employees} x 100 = {attrition_rate:.2f}%")

# ═══════════════════════════════════════════════════════
# 1. DEMOGRAPHICS
# ═══════════════════════════════════════════════════════
print_section("1a. DEMOGRAPHICS — AGE")

# Age bins
bins   = [17, 24, 34, 44, 54, 65]
labels = ["18-24", "25-34", "35-44", "45-54", "55-65"]
df["AgeBand"] = pd.cut(df["Age"], bins=bins, labels=labels)

print(f"\n  Age stats (all employees):")
print(f"    Min     : {df['Age'].min()}")
print(f"    Max     : {df['Age'].max()}")
print(f"    Mean    : {df['Age'].mean():.1f}")
print(f"    Median  : {df['Age'].median():.0f}")

print(f"\n  Age stats (left vs stayed):")
age_grp = df.groupby("Attrition", observed=True)["Age"].agg(["mean","median","min","max"]).round(1)
age_grp.index = ["Stayed", "Left"]
print(age_grp.to_string())

print(f"\n  Attrition by Age Band:")
print(attrition_table(df, "AgeBand").to_string())

# ── Gender ────────────────────────────────────────────────────────────────────
print_section("1b. DEMOGRAPHICS — GENDER")
print(attrition_table(df, "Gender").to_string())

# ── Marital Status ────────────────────────────────────────────────────────────
print_section("1c. DEMOGRAPHICS — MARITAL STATUS")
print(attrition_table(df, "MaritalStatus").to_string())

# ── Education ─────────────────────────────────────────────────────────────────
print_section("1d. DEMOGRAPHICS — EDUCATION LEVEL")
edu_map = {1: "1-Below College", 2: "2-College", 3: "3-Bachelor", 4: "4-Master", 5: "5-Doctor"}
df["EducationLabel"] = df["Education"].astype(int).map(edu_map)
print(attrition_table(df, "EducationLabel").to_string())

print_section("1e. DEMOGRAPHICS — EDUCATION FIELD")
print(attrition_table(df, "EducationField").to_string())

# ═══════════════════════════════════════════════════════
# 2. JOB CHARACTERISTICS
# ═══════════════════════════════════════════════════════
print_section("2a. JOB — DEPARTMENT")
print(attrition_table(df, "Department").to_string())

print_section("2b. JOB — JOB ROLE")
print(attrition_table(df, "JobRole").to_string())

print_section("2c. JOB — JOB LEVEL")
lvl_map = {1: "1-Entry", 2: "2-Junior", 3: "3-Mid", 4: "4-Senior", 5: "5-Director"}
df["JobLevelLabel"] = df["JobLevel"].astype(int).map(lvl_map)
print(attrition_table(df, "JobLevelLabel").to_string())

print_section("2d. JOB — BUSINESS TRAVEL")
print(attrition_table(df, "BusinessTravel").to_string())

# ═══════════════════════════════════════════════════════
# 3. COMPENSATION
# ═══════════════════════════════════════════════════════
print_section("3a. COMPENSATION — MONTHLY INCOME")

print(f"\n  Monthly Income stats (all employees):")
inc = df["MonthlyIncome"]
print(f"    Min     : ${inc.min():,.0f}")
print(f"    Max     : ${inc.max():,.0f}")
print(f"    Mean    : ${inc.mean():,.0f}")
print(f"    Median  : ${inc.median():,.0f}")
print(f"    Std Dev : ${inc.std():,.0f}")

print(f"\n  Monthly Income — Left vs Stayed:")
inc_grp = df.groupby("Attrition", observed=True)["MonthlyIncome"].agg(["mean","median","min","max"]).round(0)
inc_grp.index = ["Stayed", "Left"]
print(inc_grp.to_string())

print_section("3b. COMPENSATION — SALARY BAND")
# Equal-width salary bands
inc_bins   = [0, 3000, 6000, 9000, 12000, 21000]
inc_labels = ["<3k", "3k-6k", "6k-9k", "9k-12k", ">12k"]
df["SalaryBand"] = pd.cut(df["MonthlyIncome"], bins=inc_bins, labels=inc_labels)
print(attrition_table(df, "SalaryBand").to_string())

print_section("3c. COMPENSATION — PERCENT SALARY HIKE")
print(f"\n  % Salary Hike stats:")
hike = df["PercentSalaryHike"]
print(f"    Min     : {hike.min()}%")
print(f"    Max     : {hike.max()}%")
print(f"    Mean    : {hike.mean():.1f}%")
print(f"    Median  : {hike.median():.0f}%")

print(f"\n  Mean % Hike — Left vs Stayed:")
hike_grp = df.groupby("Attrition", observed=True)["PercentSalaryHike"].agg(["mean","median"]).round(2)
hike_grp.index = ["Stayed", "Left"]
print(hike_grp.to_string())

# Hike bands
hike_bins   = [10, 13, 16, 19, 26]
hike_labels = ["11-13%", "14-16%", "17-19%", "20-25%"]
df["HikeBand"] = pd.cut(df["PercentSalaryHike"], bins=hike_bins, labels=hike_labels)
print(f"\n  Attrition by Hike Band:")
print(attrition_table(df, "HikeBand").to_string())

print_section("3d. COMPENSATION — STOCK OPTION LEVEL")
sol_map = {0: "0-None", 1: "1-Low", 2: "2-Medium", 3: "3-High"}
df["StockLabel"] = df["StockOptionLevel"].astype(int).map(sol_map)
print(attrition_table(df, "StockLabel").to_string())

# ═══════════════════════════════════════════════════════
# 4. WORKPLACE CONDITIONS
# ═══════════════════════════════════════════════════════
print_section("4a. WORKPLACE — OVERTIME")
ot_map = {True: "Yes", False: "No"}
df["OvertimeLabel"] = df["OverTime"].map(ot_map)
print(attrition_table(df, "OvertimeLabel").to_string())

# Overtime breakdown as % of workforce
ot_count = df["OvertimeLabel"].value_counts()
print(f"\n  Overtime workforce share:")
print((ot_count / len(df) * 100).round(1).to_string())

print_section("4b. WORKPLACE — JOB SATISFACTION")
sat_map = {1: "1-Low", 2: "2-Medium", 3: "3-High", 4: "4-Very High"}
df["JobSatLabel"] = df["JobSatisfaction"].astype(int).map(sat_map)
print(attrition_table(df, "JobSatLabel").to_string())

print_section("4c. WORKPLACE — ENVIRONMENT SATISFACTION")
df["EnvSatLabel"] = df["EnvironmentSatisfaction"].astype(int).map(sat_map)
print(attrition_table(df, "EnvSatLabel").to_string())

print_section("4d. WORKPLACE — WORK-LIFE BALANCE")
wlb_map = {1: "1-Bad", 2: "2-Good", 3: "3-Better", 4: "4-Best"}
df["WLBLabel"] = df["WorkLifeBalance"].astype(int).map(wlb_map)
print(attrition_table(df, "WLBLabel").to_string())

print_section("4e. WORKPLACE — JOB INVOLVEMENT")
inv_map = {1: "1-Low", 2: "2-Medium", 3: "3-High", 4: "4-Very High"}
df["InvLabel"] = df["JobInvolvement"].astype(int).map(inv_map)
print(attrition_table(df, "InvLabel").to_string())

# ═══════════════════════════════════════════════════════
# 5. CAREER PROGRESSION
# ═══════════════════════════════════════════════════════
print_section("5a. CAREER — YEARS AT COMPANY")
yac_bins   = [-1, 1, 3, 5, 10, 20, 41]
yac_labels = ["0-1", "2-3", "4-5", "6-10", "11-20", "21+"]
df["TenureBand"] = pd.cut(df["YearsAtCompany"], bins=yac_bins, labels=yac_labels)

print(f"\n  YearsAtCompany stats:")
yac = df["YearsAtCompany"]
print(f"    Min     : {yac.min()}")
print(f"    Max     : {yac.max()}")
print(f"    Mean    : {yac.mean():.1f}")
print(f"    Median  : {yac.median():.0f}")

print(f"\n  Mean YearsAtCompany — Left vs Stayed:")
yac_grp = df.groupby("Attrition", observed=True)["YearsAtCompany"].agg(["mean","median"]).round(1)
yac_grp.index = ["Stayed", "Left"]
print(yac_grp.to_string())

print(f"\n  Attrition by Tenure Band:")
print(attrition_table(df, "TenureBand").to_string())

print_section("5b. CAREER — YEARS IN CURRENT ROLE")
print(f"\n  Mean YearsInCurrentRole — Left vs Stayed:")
ycr_grp = df.groupby("Attrition", observed=True)["YearsInCurrentRole"].agg(["mean","median"]).round(1)
ycr_grp.index = ["Stayed", "Left"]
print(ycr_grp.to_string())

ycr_bins   = [-1, 0, 2, 5, 10, 19]
ycr_labels = ["0", "1-2", "3-5", "6-10", "11+"]
df["RoleBand"] = pd.cut(df["YearsInCurrentRole"], bins=ycr_bins, labels=ycr_labels)
print(f"\n  Attrition by Years in Current Role:")
print(attrition_table(df, "RoleBand").to_string())

print_section("5c. CAREER — YEARS SINCE LAST PROMOTION")
print(f"\n  Mean YearsSinceLastPromotion — Left vs Stayed:")
prom_grp = df.groupby("Attrition", observed=True)["YearsSinceLastPromotion"].agg(["mean","median"]).round(1)
prom_grp.index = ["Stayed", "Left"]
print(prom_grp.to_string())

prom_bins   = [-1, 0, 1, 3, 7, 16]
prom_labels = ["0", "1", "2-3", "4-7", "8+"]
df["PromBand"] = pd.cut(df["YearsSinceLastPromotion"], bins=prom_bins, labels=prom_labels)
print(f"\n  Attrition by Years Since Promotion:")
print(attrition_table(df, "PromBand").to_string())

print_section("5d. CAREER — YEARS WITH CURRENT MANAGER")
print(f"\n  Mean YearsWithCurrManager — Left vs Stayed:")
mgr_grp = df.groupby("Attrition", observed=True)["YearsWithCurrManager"].agg(["mean","median"]).round(1)
mgr_grp.index = ["Stayed", "Left"]
print(mgr_grp.to_string())

mgr_bins   = [-1, 0, 2, 5, 10, 18]
mgr_labels = ["0", "1-2", "3-5", "6-10", "11+"]
df["MgrBand"] = pd.cut(df["YearsWithCurrManager"], bins=mgr_bins, labels=mgr_labels)
print(f"\n  Attrition by Years with Manager:")
print(attrition_table(df, "MgrBand").to_string())

# ═══════════════════════════════════════════════════════
# 6. CORRELATION — NUMERIC FEATURES vs ATTRITION
# ═══════════════════════════════════════════════════════
print_section("6. CORRELATION — NUMERIC FEATURES vs ATTRITION")

numeric_features = [
    "Age", "DailyRate", "DistanceFromHome", "HourlyRate",
    "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
    "PercentSalaryHike", "TotalWorkingYears", "TrainingTimesLastYear",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]
corr = df[numeric_features + ["Attrition"]].copy()
corr["Attrition"] = corr["Attrition"].astype(int)
corr_series = corr.corr()["Attrition"].drop("Attrition").sort_values()
print(f"\n  Point-biserial correlations with Attrition (sorted):")
print(corr_series.round(4).to_string())

# ═══════════════════════════════════════════════════════
# 7. KEY RISK FACTORS SUMMARY
# ═══════════════════════════════════════════════════════
print_section("7. KEY RISK FACTORS SUMMARY")

print("""
  Highest-attrition groups identified:

  DEMOGRAPHICS
    - Single employees vs married/divorced
    - Younger employees (18-24 age band)

  JOB
    - Sales Representatives and Laboratory Technicians
    - Entry-level (Job Level 1)
    - Frequent business travellers

  COMPENSATION
    - Lowest salary band (<$3k/month)
    - No stock options (level 0)

  WORKPLACE
    - Overtime workers
    - Low job satisfaction (score 1)
    - Low environment satisfaction (score 1)

  CAREER
    - Very early tenure (0-1 year)
    - Long time since promotion
""")

print(DIVIDER)
print("  EDA COMPLETE")
print(DIVIDER)

"""
HR Attrition - Database Setup
Creates hr_attrition.db (SQLite) and loads hr_attrition_features.csv
into an `employees` table.
"""

import sys
import sqlite3
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

CSV_FILE = "hr_attrition_features.csv"
DB_FILE  = "hr_attrition.db"
TABLE    = "employees"

# ── Load CSV ──────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_FILE)

# Normalise Attrition to integer 0/1 (easier to SUM/AVG in SQL)
df["Attrition"] = df["Attrition"].astype(str).map({"True": 1, "False": 0})
df["OverTime"]  = df["OverTime"].astype(str).map({"True": 1, "False": 0})

print(f"Loaded  : {len(df):,} rows × {len(df.columns)} columns from {CSV_FILE}")

# ── Connect and create table ──────────────────────────────────────────────────
conn   = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Drop and recreate for idempotency
cursor.execute(f"DROP TABLE IF EXISTS {TABLE}")

create_sql = """
CREATE TABLE employees (
    EmployeeNumber          INTEGER PRIMARY KEY,
    Age                     INTEGER,
    Attrition               INTEGER,   -- 1 = left, 0 = stayed
    BusinessTravel          TEXT,
    DailyRate               INTEGER,
    Department              TEXT,
    DistanceFromHome        INTEGER,
    Education               INTEGER,
    EducationField          TEXT,
    EnvironmentSatisfaction INTEGER,
    Gender                  TEXT,
    HourlyRate              INTEGER,
    JobInvolvement          INTEGER,
    JobLevel                INTEGER,
    JobRole                 TEXT,
    JobSatisfaction         INTEGER,
    MaritalStatus           TEXT,
    MonthlyIncome           INTEGER,
    MonthlyRate             INTEGER,
    NumCompaniesWorked      INTEGER,
    OverTime                INTEGER,   -- 1 = yes, 0 = no
    PercentSalaryHike       INTEGER,
    PerformanceRating       INTEGER,
    RelationshipSatisfaction INTEGER,
    StockOptionLevel        INTEGER,
    TotalWorkingYears       INTEGER,
    TrainingTimesLastYear   INTEGER,
    WorkLifeBalance         INTEGER,
    YearsAtCompany          INTEGER,
    YearsInCurrentRole      INTEGER,
    YearsSinceLastPromotion INTEGER,
    YearsWithCurrManager    INTEGER,
    SalaryBand              TEXT,
    AgeGroup                TEXT,
    TenureGroup             TEXT,
    PromotionGap            TEXT
);
"""
cursor.execute(create_sql)
conn.commit()

# ── Insert data ───────────────────────────────────────────────────────────────
df.to_sql(TABLE, conn, if_exists="append", index=False)
conn.commit()

# ── Verify ────────────────────────────────────────────────────────────────────
row_count = cursor.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
print(f"Inserted: {row_count:,} rows into table '{TABLE}'")
print(f"Database: {DB_FILE}")

# Show column list
cols = cursor.execute(f"PRAGMA table_info({TABLE})").fetchall()
print(f"\nTable schema ({len(cols)} columns):")
for c in cols:
    print(f"  {c[1]:<30} {c[2]}")

conn.close()
print(f"\nDone. Database ready at: {DB_FILE}")

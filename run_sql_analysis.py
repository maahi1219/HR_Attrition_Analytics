"""
HR Attrition - SQL Analysis Runner
Executes every query in hr_analysis.sql against hr_attrition.db
and prints labelled, formatted results.
"""

import sys
import re
import sqlite3
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

DB_FILE  = "hr_attrition.db"
SQL_FILE = "hr_analysis.sql"

pd.set_option("display.max_rows", 30)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.2f}".format)

DIVIDER = "=" * 70
SEP     = "-" * 70

conn = sqlite3.connect(DB_FILE)

# ── Parse SQL file into individual queries ────────────────────────────────────
with open(SQL_FILE, "r", encoding="utf-8") as f:
    raw = f.read()

# Split on query-header comments  "-- Q01 ..."
pattern = r"(-- Q\d+[^\n]*)\n(.*?)(?=-- Q\d+|$)"
matches = re.findall(pattern, raw, re.DOTALL)

for header, body in matches:
    body = body.strip()
    if not body:
        continue

    # Remove inline comments before running
    sql = re.sub(r"--[^\n]*", "", body).strip()
    if not sql.endswith(";"):
        sql += ";"

    print(f"\n{DIVIDER}")
    print(f"  {header.strip()}")
    print(DIVIDER)
    print(f"  SQL:\n{body}\n")
    print(SEP)

    try:
        df = pd.read_sql_query(sql, conn)
        if df.empty:
            print("  (no rows returned)")
        else:
            print(df.to_string(index=False))
    except Exception as e:
        print(f"  ERROR: {e}")

conn.close()
print(f"\n{DIVIDER}")
print("  ALL QUERIES COMPLETE")
print(DIVIDER)

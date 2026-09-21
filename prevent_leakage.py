"""
HR Attrition - Leakage-Safe Train/Test Split + Sklearn Pipeline
================================================================
Leakage categories addressed
  A. Temporal / post-event leakage
       Columns that are only knowable AFTER an employee has left
       are excluded entirely.
  B. Preprocessing leakage
       ALL preprocessing (scaling, encoding) is fitted exclusively
       on the training split and then applied to test — never the
       other way around. This is enforced by sklearn Pipelines.
  C. Duplicate-signal leakage
       Engineered bands (SalaryBand, AgeGroup, TenureGroup,
       PromotionGap) are dropped; their numeric sources are kept.
  D. Identifier leakage
       EmployeeNumber carries no generalizable signal and is dropped.

Pipeline structure
  ColumnTransformer
    ├── numeric branch   → StandardScaler
    └── nominal branch   → OneHotEncoder (handle_unknown='ignore')
  └── (estimator slot left open — filled in train_model.py)
"""

import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

sys.stdout.reconfigure(encoding="utf-8")

DIVIDER = "=" * 60
SEP     = "-" * 60
RANDOM_STATE = 42
TEST_SIZE    = 0.20

# ─────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────
print(DIVIDER)
print("1. LOAD")
print(SEP)

df = pd.read_csv("hr_attrition_features.csv")
df["Attrition"] = df["Attrition"].astype(str).map({"True": 1, "False": 0})
df["OverTime"]  = df["OverTime"].astype(str).map({"True": 1, "False": 0})

print(f"  Rows x Cols : {df.shape[0]:,} x {df.shape[1]}")

# ─────────────────────────────────────────────────────────────
# 2. LEAKAGE AUDIT — drop risky columns with explanation
# ─────────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("2. LEAKAGE AUDIT")
print(SEP)

# A. Temporal / post-event leakage
# These columns could only be recorded or finalised at the moment
# of or after departure; using them would give the model illegitimate
# foreknowledge. None appear in the IBM Watson dataset, but we document
# the check explicitly so the pipeline is safe if columns are added.
POST_EVENT_COLS = [
    # "ExitInterviewScore",    # only exists after leaving
    # "SeverancePay",          # only exists after leaving
    # "ResignationDate",       # future information
    # "LastDayWorked",         # future information
]

# B. Identifier — no generalizable signal
IDENTIFIER_COLS = ["EmployeeNumber"]

# C. Engineered bins — duplicates of numeric source columns already kept
DERIVED_BIN_COLS = ["SalaryBand", "AgeGroup", "TenureGroup", "PromotionGap"]

# D. Zero-variance constants — already dropped in cleaning but check again
CONSTANT_COLS = [
    c for c in df.columns
    if c != "Attrition" and df[c].nunique() <= 1
]

ALL_DROP = POST_EVENT_COLS + IDENTIFIER_COLS + DERIVED_BIN_COLS + CONSTANT_COLS
ALL_DROP = [c for c in ALL_DROP if c in df.columns]  # only drop if present

for col in ALL_DROP:
    reason = (
        "temporal/post-event" if col in POST_EVENT_COLS else
        "identifier"          if col in IDENTIFIER_COLS else
        "derived bin (source numeric kept)" if col in DERIVED_BIN_COLS else
        "zero-variance constant"
    )
    print(f"  DROP  {col:<30}  reason: {reason}")

df = df.drop(columns=ALL_DROP)
print(f"\n  Columns remaining : {df.shape[1]}")

# ─────────────────────────────────────────────────────────────
# 3. SEPARATE X AND y
# ─────────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("3. X / y SEPARATION")
print(SEP)

y = df["Attrition"].astype(int)
X = df.drop(columns=["Attrition"])

print(f"  X shape : {X.shape}  (features)")
print(f"  y shape : {y.shape}  (target — 1=Left, 0=Stayed)")

# ─────────────────────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT  —  before any preprocessing
# ─────────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("4. TRAIN / TEST SPLIT")
print(SEP)
print("  *** Split happens HERE — before fitting any transformer ***")
print(f"  Test size    : {TEST_SIZE:.0%}")
print(f"  Stratified   : yes  (preserves class ratio in both folds)")
print(f"  Random state : {RANDOM_STATE}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = TEST_SIZE,
    random_state = RANDOM_STATE,
    stratify     = y,          # keeps 83/17 ratio in both splits
)

print(f"\n  X_train : {X_train.shape}  |  y_train : {y_train.shape}")
print(f"  X_test  : {X_test.shape}   |  y_test  : {y_test.shape}")

for split_name, split_y in [("Train", y_train), ("Test", y_test)]:
    rate = split_y.mean() * 100
    print(f"  {split_name} attrition rate : {rate:.1f}%  "
          f"(Left={split_y.sum()}, Stayed={len(split_y)-split_y.sum()})")

# ─────────────────────────────────────────────────────────────
# 5. DEFINE COLUMN TYPES FOR PIPELINE
# ─────────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("5. COLUMN CLASSIFICATION FOR PIPELINE")
print(SEP)

# Nominal categoricals — need OneHotEncoder
NOMINAL_COLS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
]

# Binary and ordinal — already integers; scale with the numerics
# Ordinal columns treated as continuous for tree-based models
# and need scaling for distance-based models (SVM, KNN, LogReg)
NUMERIC_COLS = [c for c in X.columns if c not in NOMINAL_COLS]

print(f"  Numeric / ordinal / binary features ({len(NUMERIC_COLS)}):")
for c in NUMERIC_COLS:
    print(f"    {c}")

print(f"\n  Nominal features — one-hot encoded ({len(NOMINAL_COLS)}):")
for c in NOMINAL_COLS:
    cats = sorted(X_train[c].unique())
    print(f"    {c:<25} categories: {cats}")

# ─────────────────────────────────────────────────────────────
# 6. BUILD PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("6. PREPROCESSING PIPELINE")
print(SEP)

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler()),
])

nominal_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",  # unseen categories in test → all-zero row
        drop="first",             # remove one dummy per feature (dummy trap)
        sparse_output=False,
    )),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUMERIC_COLS),
        ("cat", nominal_transformer, NOMINAL_COLS),
    ],
    remainder="drop",   # silently drop any column not listed above
)

print("  ColumnTransformer")
print("    ├── num  : StandardScaler        →  numeric + ordinal + binary cols")
print("    └── cat  : OneHotEncoder(drop=first, handle_unknown=ignore)")
print("                                     →  nominal string cols")
print()
print("  *** preprocessor.fit() will only ever be called on X_train ***")
print("  *** preprocessor.transform() applied separately to X_test  ***")

# ─────────────────────────────────────────────────────────────
# 7. FIT PREPROCESSOR ON TRAIN ONLY — verify no test contamination
# ─────────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("7. FIT ON TRAIN — TRANSFORM BOTH")
print(SEP)

preprocessor.fit(X_train)                 # <-- learns mean/std from train ONLY
X_train_proc = preprocessor.transform(X_train)
X_test_proc  = preprocessor.transform(X_test)  # <-- applies train statistics

# Feature names after one-hot expansion
ohe_feature_names = (
    preprocessor
    .named_transformers_["cat"]
    .named_steps["onehot"]
    .get_feature_names_out(NOMINAL_COLS)
    .tolist()
)
all_feature_names = NUMERIC_COLS + ohe_feature_names

print(f"  X_train_proc shape : {X_train_proc.shape}")
print(f"  X_test_proc  shape : {X_test_proc.shape}")
print(f"  Total features after encoding : {len(all_feature_names)}")

# Verify: scaler means come ONLY from train
print(f"\n  Scaler fit on TRAIN only — sample means (first 5 numeric):")
scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
for col, mean_ in zip(NUMERIC_COLS[:5], scaler.mean_[:5]):
    print(f"    {col:<30} train mean = {mean_:.4f}")

# ─────────────────────────────────────────────────────────────
# 8. LEAKAGE VERIFICATION
# ─────────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("8. LEAKAGE VERIFICATION")
print(SEP)

checks = [
    ("Preprocessor fitted on train only",
     "preprocessor.fit() called once — on X_train"),
    ("Test set never seen during fit",
     "X_test passed only to .transform(), not .fit_transform()"),
    ("Stratified split",
     f"stratify=y preserves {y_train.mean()*100:.1f}% / {y_test.mean()*100:.1f}% class ratio"),
    ("No post-event columns",
     f"Checked and removed: {POST_EVENT_COLS if POST_EVENT_COLS else 'none present'}"),
    ("Identifier dropped",
     "EmployeeNumber removed before split"),
    ("Derived bins dropped",
     "SalaryBand / AgeGroup / TenureGroup / PromotionGap removed; numeric sources kept"),
    ("Unknown categories handled",
     "handle_unknown='ignore' → unseen test categories map to zero vector"),
]

for check, detail in checks:
    print(f"  [OK] {check}")
    print(f"       {detail}")

# ─────────────────────────────────────────────────────────────
# 9. SAVE ARTEFACTS
# ─────────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("9. SAVE ARTEFACTS")
print(SEP)

# Save processed arrays as CSVs (with feature names)
pd.DataFrame(X_train_proc, columns=all_feature_names).to_csv("X_train.csv", index=False)
pd.DataFrame(X_test_proc,  columns=all_feature_names).to_csv("X_test.csv",  index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv",   index=False)

# Save the fitted preprocessor for reuse in model scripts
joblib.dump(preprocessor, "preprocessor.pkl")

print("  X_train.csv        saved")
print("  X_test.csv         saved")
print("  y_train.csv        saved")
print("  y_test.csv         saved")
print("  preprocessor.pkl   saved  (fitted on train only)")

print(f"\n{DIVIDER}")
print("PIPELINE READY")
print(f"  Train : {X_train_proc.shape[0]:,} rows x {X_train_proc.shape[1]} features")
print(f"  Test  : {X_test_proc.shape[0]:,} rows  x {X_test_proc.shape[1]} features")
print(DIVIDER)

# HR Employee Attrition Analysis

## Project Description

A full end-to-end data analytics and machine learning pipeline that identifies the key drivers of employee attrition using IBM's HR Analytics dataset. The project progresses through every stage of a professional data science workflow: data auditing, cleaning, exploratory data analysis, statistical hypothesis testing, feature engineering, SQL analytics, and ML dataset preparation with leakage-safe pipelines.

**Attrition rate in dataset: 16.12%** (237 of 1,470 employees left)

---

## Dataset

| Property | Detail |
|---|---|
| **Name** | IBM HR Analytics Employee Attrition & Performance |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |
| **File** | `WA_Fn-UseC_-HR-Employee-Attrition.csv` |
| **Rows** | 1,470 |
| **Columns** | 35 |
| **Target** | `Attrition` (Yes / No) |
| **License** | Public / Open |

---

## Technologies Used

| Category | Tool / Library |
|---|---|
| Language | Python 3.10+ |
| Data manipulation | pandas, numpy |
| Statistical testing | scipy |
| Machine learning | scikit-learn |
| Model persistence | joblib |
| Database | SQLite (via Python `sqlite3`) |
| Notebook | Jupyter |

---

## Project Structure

```
HR_Attrition/
├── WA_Fn-UseC_-HR-Employee-Attrition.csv   # Raw dataset
│
├── data_audit.py                            # Step 1: Data audit report
├── data_cleaning.py                         # Step 2: Data cleaning + validation
├── eda.py                                   # Step 3: Exploratory data analysis
├── feature_engineering.py                   # Step 4: Feature creation
├── hypothesis_testing.py                    # Step 5: Statistical hypothesis testing
│
├── db_setup.py                              # Step 6: Build SQLite database
├── hr_analysis.sql                          # Step 6: 28 SQL analysis queries
├── run_sql_analysis.py                      # Step 6: SQL query runner
│
├── build_ml_dataset.py                      # Step 7: Build ML-ready dataset
├── prevent_leakage.py                       # Step 8: Leakage-safe pipeline + split
│
├── hr_attrition_clean.csv                   # Output: cleaned dataset
├── hr_attrition_features.csv               # Output: dataset with engineered features
├── hr_attrition.db                         # Output: SQLite database
├── ml_dataset.csv                          # Output: encoded ML dataset
├── X_train.csv / X_test.csv               # Output: train/test features
├── y_train.csv / y_test.csv               # Output: train/test labels
├── preprocessor.pkl                        # Output: fitted sklearn preprocessor
│
├── StudentName_HR_Attrition.ipynb          # Submission: full Jupyter Notebook
├── requirements.txt                        # Submission: dependencies
└── README.md                               # Submission: this file
```

---

## Setup & Run Instructions

### 1. Clone / download the project

```bash
git clone https://github.com/yourname/HR_Attrition.git
cd HR_Attrition
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the pipeline in order

```bash
python data_audit.py            # Audit raw dataset
python data_cleaning.py         # Clean & validate
python eda.py                   # Exploratory analysis
python feature_engineering.py   # Engineer features
python hypothesis_testing.py    # Statistical tests
python db_setup.py              # Build SQLite DB
python run_sql_analysis.py      # Run 28 SQL queries
python build_ml_dataset.py      # Encode for ML
python prevent_leakage.py       # Split + pipeline
```

Or open `StudentName_HR_Attrition.ipynb` in Jupyter and run all cells.

---

## Key Findings

### Overall
- Attrition rate: **16.12%** (237 employees left out of 1,470)

### Top Risk Factors (all statistically significant at α = 0.05)

| Factor | High-Risk Group | Attrition Rate |
|---|---|---|
| Job Involvement | Low (score 1) | 33.7% |
| Work-Life Balance | Bad (score 1) | 31.2% |
| Overtime | Yes | 30.5% |
| Job Role | Sales Representative | 39.8% |
| Tenure | 0–1 year | 34.9% |
| Age Group | 18–25 | 35.8% |
| Marital Status | Single | 25.5% |
| Salary Band | Low (<$3,632/mo) | 26.1% |
| Stock Options | None (level 0) | 24.4% |
| Business Travel | Frequent | 24.9% |

### High-Risk Profile (SQL Q22)
Employees who are **entry-level + working overtime + low job satisfaction** have a **58.7% attrition rate** — nearly 4× the company average.

### Hypothesis Testing
All 10 null hypotheses rejected at α = 0.05 using Chi-squared (categorical) and Mann-Whitney U (continuous) tests.

---

## ML Pipeline

- **Target**: `Attrition` (1 = Left, 0 = Stayed)
- **Features**: 44 after one-hot encoding
- **Split**: 80% train / 20% test, stratified
- **Preprocessing**: `StandardScaler` (numeric) + `OneHotEncoder` (nominal), fitted on train only
- **Class imbalance**: 5.2:1 — use `class_weight='balanced'` or SMOTE at modelling stage

---

## Author

MAAHI SHARMA   


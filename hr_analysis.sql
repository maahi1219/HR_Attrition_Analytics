-- =============================================================
-- HR Attrition — SQL Analysis
-- Database : hr_attrition.db  (SQLite)
-- Table    : employees
-- =============================================================
-- Sections
--   A. Basic        Q01–Q06
--   B. Intermediate Q07–Q15
--   C. Advanced     Q16–Q28
-- =============================================================


-- ─────────────────────────────────────────────────────────────
-- A. BASIC
-- ─────────────────────────────────────────────────────────────

-- Q01 Total employees
SELECT COUNT(*) AS total_employees
FROM employees;

-- Q02 Employees who left vs stayed
SELECT
    CASE WHEN Attrition = 1 THEN 'Left' ELSE 'Stayed' END AS status,
    COUNT(*) AS count
FROM employees
GROUP BY Attrition;

-- Q03 Overall attrition rate
SELECT
    COUNT(*)                                         AS total_employees,
    SUM(Attrition)                                   AS employees_left,
    ROUND(SUM(Attrition) * 100.0 / COUNT(*), 2)     AS attrition_rate_pct
FROM employees;

-- Q04 Average, minimum, and maximum monthly income
SELECT
    ROUND(AVG(MonthlyIncome), 2)  AS avg_income,
    MIN(MonthlyIncome)            AS min_income,
    MAX(MonthlyIncome)            AS max_income
FROM employees;

-- Q05 Headcount by department
SELECT
    Department,
    COUNT(*) AS headcount
FROM employees
GROUP BY Department
ORDER BY headcount DESC;

-- Q06 Headcount and gender split
SELECT
    Gender,
    COUNT(*)                                        AS count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM employees), 1) AS pct
FROM employees
GROUP BY Gender
ORDER BY count DESC;


-- ─────────────────────────────────────────────────────────────
-- B. INTERMEDIATE
-- ─────────────────────────────────────────────────────────────

-- Q07 Attrition by department
SELECT
    Department,
    COUNT(*)                                             AS total,
    SUM(Attrition)                                       AS left_count,
    ROUND(SUM(Attrition) * 100.0 / COUNT(*), 1)         AS attrition_pct
FROM employees
GROUP BY Department
ORDER BY attrition_pct DESC;

-- Q08 Attrition by job role
SELECT
    JobRole,
    COUNT(*)                                             AS total,
    SUM(Attrition)                                       AS left_count,
    ROUND(SUM(Attrition) * 100.0 / COUNT(*), 1)         AS attrition_pct
FROM employees
GROUP BY JobRole
ORDER BY attrition_pct DESC;

-- Q09 Attrition by salary band
SELECT
    SalaryBand,
    COUNT(*)                                             AS total,
    SUM(Attrition)                                       AS left_count,
    ROUND(SUM(Attrition) * 100.0 / COUNT(*), 1)         AS attrition_pct
FROM employees
GROUP BY SalaryBand
ORDER BY CASE SalaryBand WHEN 'Low' THEN 1 WHEN 'Medium' THEN 2 WHEN 'High' THEN 3 END;

-- Q10 Attrition by overtime status
SELECT
    CASE WHEN OverTime = 1 THEN 'Yes' ELSE 'No' END      AS overtime,
    COUNT(*)                                              AS total,
    SUM(Attrition)                                        AS left_count,
    ROUND(SUM(Attrition) * 100.0 / COUNT(*), 1)          AS attrition_pct
FROM employees
GROUP BY OverTime
ORDER BY attrition_pct DESC;

-- Q11 Attrition by marital status
SELECT
    MaritalStatus,
    COUNT(*)                                             AS total,
    SUM(Attrition)                                       AS left_count,
    ROUND(SUM(Attrition) * 100.0 / COUNT(*), 1)         AS attrition_pct
FROM employees
GROUP BY MaritalStatus
ORDER BY attrition_pct DESC;

-- Q12 Attrition by business travel frequency
SELECT
    BusinessTravel,
    COUNT(*)                                             AS total,
    SUM(Attrition)                                       AS left_count,
    ROUND(SUM(Attrition) * 100.0 / COUNT(*), 1)         AS attrition_pct
FROM employees
GROUP BY BusinessTravel
ORDER BY attrition_pct DESC;

-- Q13 Attrition by age group
SELECT
    AgeGroup,
    COUNT(*)                                             AS total,
    SUM(Attrition)                                       AS left_count,
    ROUND(SUM(Attrition) * 100.0 / COUNT(*), 1)         AS attrition_pct
FROM employees
GROUP BY AgeGroup
ORDER BY CASE AgeGroup
    WHEN '18-25' THEN 1 WHEN '26-35' THEN 2 WHEN '36-45' THEN 3
    WHEN '46-55' THEN 4 WHEN '56+'   THEN 5 END;

-- Q14 Attrition by tenure group
SELECT
    TenureGroup,
    COUNT(*)                                             AS total,
    SUM(Attrition)                                       AS left_count,
    ROUND(SUM(Attrition) * 100.0 / COUNT(*), 1)         AS attrition_pct
FROM employees
GROUP BY TenureGroup
ORDER BY CASE TenureGroup
    WHEN '0-1' THEN 1 WHEN '2-3' THEN 2 WHEN '4-5' THEN 3
    WHEN '6-10' THEN 4 WHEN '10+' THEN 5 END;

-- Q15 Average income by department and attrition status
SELECT
    Department,
    CASE WHEN Attrition = 1 THEN 'Left' ELSE 'Stayed' END  AS status,
    ROUND(AVG(MonthlyIncome), 0)                            AS avg_income,
    COUNT(*)                                                AS headcount
FROM employees
GROUP BY Department, Attrition
ORDER BY Department, Attrition;


-- ─────────────────────────────────────────────────────────────
-- C. ADVANCED
-- ─────────────────────────────────────────────────────────────

-- Q16 CTE — Attrition rate per job role vs company average
WITH role_stats AS (
    SELECT
        JobRole,
        COUNT(*)                                             AS total,
        SUM(Attrition)                                       AS left_count,
        ROUND(SUM(Attrition) * 100.0 / COUNT(*), 1)         AS role_attrition_pct
    FROM employees
    GROUP BY JobRole
),
company_avg AS (
    SELECT ROUND(SUM(Attrition) * 100.0 / COUNT(*), 1) AS overall_pct
    FROM employees
)
SELECT
    r.JobRole,
    r.total,
    r.left_count,
    r.role_attrition_pct,
    c.overall_pct                                            AS company_avg_pct,
    ROUND(r.role_attrition_pct - c.overall_pct, 1)          AS deviation_from_avg
FROM role_stats r
CROSS JOIN company_avg c
ORDER BY deviation_from_avg DESC;

-- Q17 CASE bucketing — income quartile labels and attrition
SELECT
    CASE
        WHEN MonthlyIncome <= 2911  THEN 'Q1 Bottom 25%'
        WHEN MonthlyIncome <= 4919  THEN 'Q2 Lower-Mid'
        WHEN MonthlyIncome <= 8379  THEN 'Q3 Upper-Mid'
        ELSE                             'Q4 Top 25%'
    END                                                      AS income_quartile,
    COUNT(*)                                                 AS total,
    SUM(Attrition)                                           AS left_count,
    ROUND(SUM(Attrition) * 100.0 / COUNT(*), 1)             AS attrition_pct
FROM employees
GROUP BY income_quartile
ORDER BY income_quartile;

-- Q18 Subquery — employees earning below the department average
SELECT
    e.EmployeeNumber,
    e.Department,
    e.JobRole,
    e.MonthlyIncome,
    ROUND(dept_avg.avg_income, 0)                            AS dept_avg_income,
    CASE WHEN e.Attrition = 1 THEN 'Left' ELSE 'Stayed' END AS status
FROM employees e
JOIN (
    SELECT Department, AVG(MonthlyIncome) AS avg_income
    FROM employees
    GROUP BY Department
) dept_avg ON e.Department = dept_avg.Department
WHERE e.MonthlyIncome < dept_avg.avg_income
ORDER BY e.Department, e.MonthlyIncome;

-- Q19 Window function — top 5 income rank within each department
--     (window functions cannot appear in WHERE; wrap in a derived table)
SELECT *
FROM (
    SELECT
        EmployeeNumber,
        Department,
        JobRole,
        MonthlyIncome,
        RANK() OVER (PARTITION BY Department ORDER BY MonthlyIncome DESC) AS income_rank_in_dept,
        CASE WHEN Attrition = 1 THEN 'Left' ELSE 'Stayed' END            AS status
    FROM employees
) ranked
WHERE income_rank_in_dept <= 5
ORDER BY Department, income_rank_in_dept;

-- Q20 Window function — running total of attrition by tenure
SELECT
    TenureGroup,
    SUM(Attrition)                                                       AS left_in_band,
    SUM(SUM(Attrition)) OVER (
        ORDER BY CASE TenureGroup
            WHEN '0-1' THEN 1 WHEN '2-3' THEN 2 WHEN '4-5' THEN 3
            WHEN '6-10' THEN 4 WHEN '10+' THEN 5 END
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )                                                                    AS running_total_left,
    COUNT(*)                                                             AS headcount
FROM employees
GROUP BY TenureGroup
ORDER BY CASE TenureGroup
    WHEN '0-1' THEN 1 WHEN '2-3' THEN 2 WHEN '4-5' THEN 3
    WHEN '6-10' THEN 4 WHEN '10+' THEN 5 END;

-- Q21 Window function — income percentile per employee
SELECT
    EmployeeNumber,
    JobRole,
    MonthlyIncome,
    ROUND(
        PERCENT_RANK() OVER (ORDER BY MonthlyIncome) * 100, 1
    )                                                                    AS income_percentile,
    CASE WHEN Attrition = 1 THEN 'Left' ELSE 'Stayed' END               AS status
FROM employees
ORDER BY income_percentile DESC;

-- Q22 CTE — high-risk employee profile
--     Defines "high risk" as: entry-level + overtime + low job satisfaction
WITH high_risk AS (
    SELECT *
    FROM employees
    WHERE JobLevel = 1
      AND OverTime = 1
      AND JobSatisfaction <= 2
)
SELECT
    COUNT(*)                                                 AS high_risk_count,
    SUM(Attrition)                                           AS left_count,
    ROUND(SUM(Attrition) * 100.0 / COUNT(*), 1)             AS attrition_pct,
    ROUND(AVG(MonthlyIncome), 0)                             AS avg_income,
    ROUND(AVG(YearsAtCompany), 1)                            AS avg_tenure
FROM high_risk;

-- Q23 CTE + window — department attrition rank
WITH dept_attrition AS (
    SELECT
        Department,
        COUNT(*)                                             AS total,
        SUM(Attrition)                                       AS left_count,
        ROUND(SUM(Attrition) * 100.0 / COUNT(*), 1)         AS attrition_pct
    FROM employees
    GROUP BY Department
)
SELECT
    Department,
    total,
    left_count,
    attrition_pct,
    RANK() OVER (ORDER BY attrition_pct DESC)                AS attrition_rank
FROM dept_attrition;

-- Q24 Multi-factor cross-tab — overtime × marital status attrition
SELECT
    CASE WHEN OverTime = 1 THEN 'Yes' ELSE 'No' END         AS overtime,
    MaritalStatus,
    COUNT(*)                                                 AS total,
    SUM(Attrition)                                           AS left_count,
    ROUND(SUM(Attrition) * 100.0 / COUNT(*), 1)             AS attrition_pct
FROM employees
GROUP BY OverTime, MaritalStatus
ORDER BY attrition_pct DESC;

-- Q25 Subquery — roles where attrition rate exceeds company average
SELECT
    JobRole,
    total,
    left_count,
    attrition_pct
FROM (
    SELECT
        JobRole,
        COUNT(*)                                             AS total,
        SUM(Attrition)                                       AS left_count,
        ROUND(SUM(Attrition) * 100.0 / COUNT(*), 1)         AS attrition_pct
    FROM employees
    GROUP BY JobRole
) role_rates
WHERE attrition_pct > (
    SELECT ROUND(SUM(Attrition) * 100.0 / COUNT(*), 1) FROM employees
)
ORDER BY attrition_pct DESC;

-- Q26 Window + CASE — flag each employee's income vs their role average
--     (SQLite has no MEDIAN window function; AVG used as a proxy)
SELECT
    EmployeeNumber,
    JobRole,
    MonthlyIncome,
    ROUND(
        AVG(MonthlyIncome) OVER (PARTITION BY JobRole), 0
    )                                                                    AS role_avg_income,
    CASE
        WHEN MonthlyIncome < AVG(MonthlyIncome) OVER (PARTITION BY JobRole)
            THEN 'Below Average'
        WHEN MonthlyIncome > AVG(MonthlyIncome) OVER (PARTITION BY JobRole)
            THEN 'Above Average'
        ELSE 'At Average'
    END                                                                  AS income_vs_avg,
    CASE WHEN Attrition = 1 THEN 'Left' ELSE 'Stayed' END               AS status
FROM employees
ORDER BY JobRole, MonthlyIncome;

-- Q27 CTE chain — satisfaction index and attrition
--     Composite score = average of JobSatisfaction + EnvironmentSatisfaction + WorkLifeBalance
WITH satisfaction AS (
    SELECT
        EmployeeNumber,
        Attrition,
        ROUND((JobSatisfaction + EnvironmentSatisfaction + WorkLifeBalance) / 3.0, 2) AS satisfaction_index
    FROM employees
),
banded AS (
    SELECT
        CASE
            WHEN satisfaction_index < 2   THEN '1 Low    (<2.0)'
            WHEN satisfaction_index < 3   THEN '2 Medium (2.0–2.9)'
            WHEN satisfaction_index < 3.5 THEN '3 High   (3.0–3.4)'
            ELSE                               '4 Very High (3.5+)'
        END                                                              AS sat_band,
        Attrition
    FROM satisfaction
)
SELECT
    sat_band,
    COUNT(*)                                                             AS total,
    SUM(Attrition)                                                       AS left_count,
    ROUND(SUM(Attrition) * 100.0 / COUNT(*), 1)                         AS attrition_pct
FROM banded
GROUP BY sat_band
ORDER BY sat_band;

-- Q28 Window function — top 3 highest-paid leavers per department
SELECT *
FROM (
    SELECT
        EmployeeNumber,
        Department,
        JobRole,
        MonthlyIncome,
        YearsAtCompany,
        RANK() OVER (PARTITION BY Department ORDER BY MonthlyIncome DESC) AS pay_rank
    FROM employees
    WHERE Attrition = 1
)
WHERE pay_rank <= 3
ORDER BY Department, pay_rank;

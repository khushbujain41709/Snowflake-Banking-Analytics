# Banking Customer Analytics & Segmentation

> An end-to-end banking analytics pipeline built on **Snowflake Medallion Architecture**, transforming raw customer data into actionable business intelligence and ML-powered segmentation.

**Prepared by:** Lakshita Chandrakar · Khushbu Jain 

---

## Overview

This project implements a complete data engineering and analytics solution for banking customer data. Raw CSV data flows through a structured **Bronze → Silver → Gold** pipeline, followed by a unified analytics layer, customer segmentation reporting, and a machine learning prediction model - enabling banks to classify customers, detect risk, and drive data-informed decisions.

**Dataset:** 5,000 customer records · 40+ attributes spanning accounts, transactions, loans, credit cards, and customer support.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Warehouse | Snowflake Data Cloud |
| Data Engineering | SQL (DDL + Stored Procedures) |
| Analytics & ML | Python, Pandas, Scikit-learn |
| Classification | XGBoost + MultiOutputClassifier |
| Encoding | MultiLabelBinarizer |

---

## Architecture

The pipeline follows **Medallion Architecture** for progressive data quality improvement:

```
Raw CSV Files
      │
      ▼
┌─────────────┐
│ Bronze Layer│  Raw ingestion → Snowflake stage → BANKING_DATA_RAW
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Silver Layer│  Cleaning · Validation · Anomaly Detection
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Gold Layer │  Star Schema → CUSTOMER · TRANSACTION · LOAN tables
└──────┬──────┘
       │
       ▼
┌──────────────────────┐
│  Unified Profile     │  360° customer view + Feature Engineering
└──────────┬───────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
Segmentation    ML Prediction
 Report          (predict.py)
```

---

## Pipeline Layers

### Bronze Layer - `bronze.sql`

Raw data ingestion with zero transformation. Establishes the single source of truth.

- Creates `bankDB` database and `bankSchema`
- Defines `BANKING_DATA_RAW` table (40 columns)
- Configures CSV format and internal Snowflake stage
- Loads data via `COPY INTO` and validates completeness

---

### Silver Layer - `silver.sql`

Data quality, standardization, and anomaly detection.

**Cleaning & Transformation**
- Corrects negative minimum payment values to zero
- Standardizes gender codes (M / F / O)
- Constructs `FULL_NAME` from first and last name fields
- Removes duplicate records and redundant columns

**Anomaly Detection (Stored Procedures)**

| Procedure | Method |
|---|---|
| `NULL_VALUE_ANOMALIES()` | Flags missing values in critical fields |
| `NEGATIVE_VALUE_ANOMALIES()` | Detects negative financial figures |
| `Z_SCORE_ANOMALIES()` | Statistical outliers where \|Z\| > 3 |
| Credit Utilization Check | IQR-based abnormal utilization detection |

All flags are consolidated into a single `ANOMALY` column.

---

### Gold Layer - `gold.sql`

Normalized star schema optimized for analytics and reporting.

| Table | Primary Key | Key Contents |
|---|---|---|
| `CUSTOMER` | `CUSTOMER_ID` | Demographics, contact info, account type |
| `TRANSACTION_TABLE` | `TRANSACTIONID` | Amount, type, date, post-transaction balance |
| `LOAN` | `LOAN_ID` | Loan details, credit card balance, rewards, payment due |

Foreign keys on `CUSTOMER_ID` link all domain tables.

---

### Unified Customer Profile - `unifiedCustomerProfile.sql`

Joins all Gold tables into a single analytical view and engineers business features.

**Derived Features**

```
Credit Utilization (%) = (Credit Card Balance / Credit Limit) × 100

Money Issues Ratio = (Loan Amount + Credit Card Balance) / Account Balance
```

**Segmentation Thresholds Table**: percentile-based dynamic boundaries:
- P25 / P75 / P90 Account Balance
- P75 Rewards Points
- P25 / P75 Money Issues Ratio
- Average and standard deviation of Credit Limit

---

### Segmentation Report - `segmentedReport.sql`

Rule-based classification of customers into business segments stored in `CUSTOMER_TYPE` (array - supports multi-label assignment).

| Segment | Key Criteria |
|---|---|
| **High Value** | No anomalies · Credit utilization ≤ 10% · Balance ≥ P75 · Rewards ≥ P75 · Loan approved/closed |
| **Risky** | Anomaly detected OR credit utilization ≥ 80% OR ≥ 2 moderate risk indicators |
| **Loan Eligible** | No anomalies · Utilization ≤ 30% · Active within 1 year · Balance ≥ median |
| **Warning** | No anomalies + loan rejected OR utilization 30–50% OR balance < P25 OR inactive > 1 year |
| **Normal** | Does not qualify for any of the above segments |

An **overlap analysis** identifies customers belonging to multiple segments simultaneously.

**Segment Distribution (5,000 customers)**

| Segment | Count | Share |
|---|---|---|
| Warning | 1,210 | 24.1% |
| Normal | 1,183 | 23.7% |
| Risky | 1,120 | 22.4% |
| Risky + Warning | 568 | 11.4% |
| Loan Eligible | 508 | 10.2% |
| Other combinations | ~411 | ~8.2% |

---

### Business Insights - `businessInsights.sql`

Analytical reports generated from the unified dataset:

- **Monthly Transaction Analysis** - trend detection and seasonal pattern identification
- **City-Wise Balance Analysis** - regional customer value comparison
- **Branch-Wise Loan Analysis** - branch performance and loan volume benchmarking
- **Account Type Analysis** - savings vs current distribution and balance comparison
- **Interest Rate Analysis** - loan pricing trends by loan type

**Key Metrics**

| Metric | Value |
|---|---|
| Total Customers | 5K |
| Sum of Transaction Amount | $13.34M |
| Sum of Loan Amount | $218.45M |
| Sum of Account Balance | $36.71M |
| Total Loan Count | 3.37K |

---

### Machine Learning Layer - `predict.py`

Automates customer segmentation using **multi-label XGBoost classification**.

**Pipeline**

```
Data Collection → Preprocessing → Feature Engineering → Label Encoding
      → Train/Test Split → XGBoost Training → Evaluation → Deployment
```

**Feature Engineering** - 64 total features including:
- Date-derived: account age, days since last transaction, days to payment due
- Missing value indicator flags
- One-hot encoded categoricals (account type, transaction type, loan status, card type)

**Model Configuration**

| Component | Choice |
|---|---|
| Classifier | XGBoostClassifier |
| Multi-label wrapper | MultiOutputClassifier |
| Label transformation | MultiLabelBinarizer |
| Train / Test split | 4,000 / 1,000 |

**Model Performance**

| Metric | Score |
|---|---|
| Subset Accuracy | **97.90%** |
| Hamming Loss | 0.0056 |
| Micro F1 Score | 0.9881 |
| Macro F1 Score | 0.9858 |

**Per-Segment Results**

| Segment | Precision | Recall | F1 |
|---|---|---|---|
| HIGH_VALUE | 1.00 | 0.96 | 0.98 |
| LOAN_ELIGIBLE | 0.97 | 1.00 | 0.98 |
| NORMAL | 0.99 | 0.99 | 0.99 |
| RISKY | 0.99 | 0.98 | 0.98 |
| WARNING | 1.00 | 1.00 | 1.00 |

**Top 5 Most Important Features**

| Rank | Feature | Importance |
|---|---|---|
| 1 | Credit Utilization Percentage | 0.1735 |
| 2 | Account Balance | 0.1336 |
| 3 | Anomaly Flag | 0.1135 |
| 4 | Credit Card Balance | 0.0711 |
| 5 | Loan Status (Rejected) | 0.0611 |

**Saved Artifacts**
```
models/
├── customer_classifier.pkl
├── label_encoder.pkl
├── metadata.pkl
└── feature_importance.csv
```

---

## Project Structure

```
Banking-Customer-Segmentation/
├── bronze.sql                  # Raw data ingestion
├── silver.sql                  # Cleaning & anomaly detection
├── gold.sql                    # Star schema modeling
├── unifiedCustomerProfile.sql  # 360° customer view + thresholds
├── segmentedReport.sql         # Business segmentation rules
├── businessInsights.sql        # Analytical reports
├── predict.py                  # ML training & prediction
├── models/                     # Saved model artifacts
└── README.md
```

---

## Key Business Value

- Identify and retain **high-value customers** with targeted loyalty programs
- Detect **risky customers** early for proactive risk management and fraud prevention
- Surface **loan-eligible customers** for personalized lending offers
- Flag **warning customers** for timely intervention before risk escalates
- Automate classification at scale using **ML-powered prediction**

---

## Future Enhancements

- Real-time data ingestion via **Snowflake Streams & Tasks** (CDC)
- Interactive dashboards with **Power BI or Tableau**
- **Automated model retraining** pipelines
- Real-time customer scoring and alert frameworks

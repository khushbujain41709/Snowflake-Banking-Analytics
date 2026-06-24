# Banking Customer Analytics Pipeline

## Overview

This project implements a complete **Banking Customer Analytics Pipeline** using the **Medallion Architecture (Bronze → Silver → Gold)** followed by an **Analytics Layer**, **Business Reporting Layer**, and a **Machine Learning Prediction Layer**. The objective is to transform raw banking data into meaningful business insights and predictive customer segmentation.

The pipeline begins with raw CSV data ingestion, progresses through data cleaning and transformation, creates a normalized relational model, generates customer segments based on business rules, and finally trains a machine learning model capable of predicting customer segments for new customers.

---

### Customer Segments
- High Value Customers
- Risky Customers
- Loan Eligible Customers
- Warning Customers
- Normal Customers

---

## Tech Stack
- Snowflake
- SQL
- Pandas
- Scikit Learn for XGBoostClassifier

---

## File Summary

### bronze.sql
**Purpose:** Bronze layer - Raw data ingestion and staging
- This file sets up the foundational data infrastructure:
- Creates database bankDB and schema bankSchema
- Defines the main raw table banking_data_raw with all 40 columns including customer info, transactions, loans, credit cards, and feedback
- Establishes a CSV file format with proper delimiters and date formatting
- Creates a stage(bank_stage) for data loading
- Copies data from the stage into the raw table
- Displays final data for verification
- Key actions: Infrastructure setup, schema definition, raw data ingestion

---

### silver.sql
**Purpose:** Silver layer - Data cleaning, transformation, and anomaly detection
- This file handles data quality and ETL operations:
- NULL value detection: Identifies missing values across all columns using multiple methods
- Data cleaning:
    - Corrects negative minimum payment values to zero
    - Standardizes gender values (M/F/O)
    - Creates FULL_NAME by concatenating first and last names
    - Removes duplicate rows
    - Drops redundant columns (FIRST_NAME, LAST_NAME)
- Anomaly detection procedures:
    - NULL_VALUE_ANOMALIES(): Flags NULL values in key columns
    - NEGATIVE_VALUE_ANOMALIES(): Flags negative values in numeric columns
    - Z_SCORE_ANOMALIES(): Identifies statistical outliers using z-score (>3)
    - Credit utilization anomaly detection using IQR method
- Final anomaly flag: Combines all anomaly flags into a single ANOMALY column
- Data validation: Checks for invalid ages (<18 or >120) and negative transactions
- Key actions: Data quality improvement, standardization, anomaly detection, outlier identification

---

### gold.sql
**Purpose:** Gold layer - Data modeling and dimensional structuring
- This file transforms raw data into a normalized star schema:
- Creates three dimension/fact tables:
    - CUSTOMER: Core customer information (demographics, contact details)
    - TRANSACTION_TABLE: Transaction history (amounts, dates, types)
    - LOAN: Loan details, credit card info, and payment information
- Establishes relationships:
    - Primary keys on CUSTOMER_ID, TRANSACTIONID, and LOAN_ID
    - Foreign keys linking transactions and loans back to customers
- Data validation: Counts records in each table to ensure data integrity
- Key actions: Data normalization, table creation, relationship establishment, integrity validation

---

### unifiedCustomerProfile.sql
**Purpose:** Analytics layer - Unified view and segmentation
- This file creates a comprehensive analytical dataset:
- UNIFIED_BANK_DATA: Joins all three gold tables (Customer + Transaction + Loan) to create a single view
- Feature engineering:
    - Calculates CREDIT_UTILIZATION_PERCENTAGE (credit card balance / credit limit × 100)
- Customer segmentation thresholds:
    - Creates CUSTOMER_SEGMENT_THRESHOLDS table with percentile values:
        - P25 and P75 for account balance
        - P75 for rewards points
        - P25 and P75 for money issues ratio (loan amount / account balance)
- Unified analysis ready: Combines all customer data with derived metrics for advanced analytics

---

### segmentedReport.sql
**Purpose:** Reporting layer - Customer segmentation and risk analysis
- This file creates strategic customer segments for business decision-making:
- Segment 1: HIGH_VALUE_CUSTOMERS (Low risk, high value)
    - No anomalies detected
    - Credit utilization ≤ 10%
    - Account balance ≥ 75th percentile
    - Rewards points ≥ 75th percentile
    - Loan status is approved or closed
- Segment 2: RISKY_CUSTOMERS (High risk indicators)
    - Has anomalies OR
    - Credit utilization ≥ 50% OR
    - Minimum payment due > 0.5 + (5% of balance) OR
    - Money issues ratio exceeds upper whisker (Q3 + 1.5×IQR)
- Segment 3: LOAN_ELIGIBLE (Good candidates for loans)
    - No anomalies
    - Credit utilization ≤ 30%
    - Loan status approved/closed
    - Active in last year
    - Account balance ≥ median
- Segment 4: WARNING_CUSTOMERS (Needs attention)
    - No anomalies AND any of:
    - Loan rejected OR
    - Moderate credit utilization (30-50%) OR
    - Account balance < 25th percentile OR
    - Inactive (>1 year since transaction)
- Overlap analysis: Cross-checks between segments to identify customers falling into multiple categories

---

## Business Insights
The pipeline helps banks:
- Identify premium customers
- Detect high-risk customers
- Recommend loan-eligible customers
- Improve customer targeting and retention
- Support data-driven business decisions

---

## Project Structure

```
Banking-Customer-Segmentation
│── bronze.sql
│── silver.sql
│── gold.sql
│── unifiedCustomerProfile.sql
│── segmentedReport.sql
└── README.md
```

---

## Future Enhancements
- Interactive Power BI Dashboard
- Automated Snowflake Tasks & Streams
- PySpark-based ETL Pipeline
- ML-based Customer Risk Prediction

---

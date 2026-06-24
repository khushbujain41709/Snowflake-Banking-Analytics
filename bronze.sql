USE ROLE ACCOUNTADMIN;
USE WAREHOUSE BANKING_A;

CREATE OR REPLACE DATABASE bankDB;
CREATE OR REPLACE SCHEMA bankSchema;

SELECT CURRENT_DATABASE(), CURRENT_SCHEMA();

CREATE OR REPLACE TABLE banking_data_raw(
    Customer_ID INTEGER,
    First_Name VARCHAR,
    Last_Name VARCHAR,
    Age INTEGER,
    Gender VARCHAR,
    Address VARCHAR,
    City VARCHAR,
    Contact_Number VARCHAR,
    Email VARCHAR,
    Account_Type VARCHAR,
    Account_Balance NUMBER(15,2),
    Date_Of_Account_Opening DATE,
    Last_Transaction_Date DATE,
    TransactionID INTEGER,
    Transaction_Date DATE,
    Transaction_Type VARCHAR,
    Transaction_Amount NUMBER(15,2),
    Account_Balance_After_Transaction NUMBER(15,2),
    Branch_ID INTEGER,
    Loan_ID INTEGER,
    Loan_Amount NUMBER(15,2),
    Loan_Type VARCHAR,
    Interest_Rate NUMBER(8,4),
    Loan_Term INTEGER,
    Approval_Rejection_Date DATE,
    Loan_Status VARCHAR,
    CardID INTEGER,
    Card_Type VARCHAR,
    Credit_Limit NUMBER(15,2),
    Credit_Card_Balance NUMBER(15,2),
    Minimum_Payment_Due NUMBER(15,2),
    Payment_Due_Date DATE,
    Last_Credit_Card_Payment_Date DATE,
    Rewards_Points INTEGER,
    Feedback_ID INTEGER,
    Feedback_Date DATE,
    Feedback_Type VARCHAR,
    Resolution_Status VARCHAR,
    Resolution_Date DATE,
    Anomaly INTEGER
);

SHOW TABLES;

// File format helps us to match column name - it automatically matches the columns by checking order. It handles the space between names and comma id csv is used and also enclose string by double quotes.
CREATE OR REPLACE FILE FORMAT csv_format
TYPE = 'CSV'
RECORD_DELIMITER = '\n'
FIELD_DELIMITER = ','
FIELD_OPTIONALLY_ENCLOSED_BY = '"'
SKIP_HEADER = 1
DATE_FORMAT = 'MM/DD/YYYY';

SHOW STAGES;
CREATE OR REPLACE STAGE bank_stage;
COPY INTO banking_data_raw from @bank_stage FILE_FORMAT = csv_format;

SELECT * FROM BANKING_DATA_RAW;
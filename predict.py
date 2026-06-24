# Target column is multi-label, not multi-class as CUSTOMER_TYPE is an ARRAY.
# We can use XGBoost + MultiOutputClassifier using MultiLableBinarizer()
"""
===========================================================
Banking Customer Segmentation Prediction
-----------------------------------------------------------
Author : Khushbu Jain and Lakshita Chandrakar
Model  : Multi-label Customer Segmentation
Algorithm : XGBoost + MultiOutputClassifier

Predicts:
- HIGH_VALUE
- RISKY
- LOAN_ELIGIBLE
- WARNING
- NORMAL
===========================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import ast
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import(
    OneHotEncoder,
    MultiLabelBinarizer
)
from sklearn.metrics import(
    classification_report,
    accuracy_score,
    hamming_loss,
    f1_score
)
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

# Paths

DATA_PATH = "unified_bank_data.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "customer_classifier.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

# Load Dataset
df = pd.read_csv(DATA_PATH)
print("Dataset Shape :", df.shape)
print()
print(df.head())

# Remove Duplicate Customers
df = df.drop_duplicates(subset=["CUSTOMER_ID"])
print("After Removing Duplicates :", df.shape)

# Convert CUSTOMER_TYPE to List
def parse_labels(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except:
        return [i.strip() for i in str(x).split(",")]

df["CUSTOMER_TYPE"] = df["CUSTOMER_TYPE"].apply(parse_labels)

# Date Feature Engineering
DATE_COLUMNS = [
    "DATE_OF_ACCOUNT_OPENING",
    "LAST_TRANSACTION_DATE",
    "TRANSACTION_DATE",
    "APPROVAL_REJECTION_DATE",
    "PAYMENT_DUE_DATE",
    "LAST_CREDIT_CARD_PAYMENT_DATE"
]

def create_date_features(df):
    df = df.copy()
    today = pd.Timestamp.today().normalize()

    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

            df[col + "_IS_MISSING"] = df[col].isna().astype(int)
            df[col + "_YEAR"] = df[col].dt.year
            df[col + "_MONTH"] = df[col].dt.month
            df[col + "_DAY"] = df[col].dt.day
            df[col + "_DAYOFWEEK"] = df[col].dt.dayofweek

    if "DATE_OF_ACCOUNT_OPENING" in df.columns:
        df["ACCOUNT_AGE_DAYS"] = (
            today - df["DATE_OF_ACCOUNT_OPENING"]
        ).dt.days

    if "LAST_TRANSACTION_DATE" in df.columns:
        df["DAYS_SINCE_LAST_TRANSACTION"] = (
            today - df["LAST_TRANSACTION_DATE"]
        ).dt.days

    if "PAYMENT_DUE_DATE" in df.columns:
        df["DAYS_TO_PAYMENT_DUE"] = (
            df["PAYMENT_DUE_DATE"] - today
        ).dt.days

    if "LAST_CREDIT_CARD_PAYMENT_DATE" in df.columns:
        df["DAYS_SINCE_LAST_CARD_PAYMENT"] = (
            today - df["LAST_CREDIT_CARD_PAYMENT_DATE"]
        ).dt.days

    df = df.drop(columns=DATE_COLUMNS, errors="ignore")

    return df
USEFUL_COLUMNS = [
    "ACCOUNT_TYPE",
    "ACCOUNT_BALANCE",
    "ANOMALY",

    "TRANSACTION_DATE",
    "TRANSACTION_TYPE",
    "TRANSACTION_AMOUNT",
    "ACCOUNT_BALANCE_AFTER_TRANSACTION",

    "LOAN_AMOUNT",
    "LOAN_TYPE",
    "INTEREST_RATE",
    "LOAN_TERM",
    "APPROVAL_REJECTION_DATE",
    "LOAN_STATUS",

    "CARD_TYPE",
    "CREDIT_LIMIT",
    "CREDIT_CARD_BALANCE",
    "MINIMUM_PAYMENT_DUE",
    "PAYMENT_DUE_DATE",
    "LAST_CREDIT_CARD_PAYMENT_DATE",
    "REWARDS_POINTS",
    "CREDIT_UTILIZATION_PERCENTAGE",

    "DATE_OF_ACCOUNT_OPENING",
    "LAST_TRANSACTION_DATE"
]

DROP_COLUMNS = [
    "CUSTOMER_ID",
    "FULL_NAME",
    "AGE",
    "GENDER",
    "ADDRESS",
    "CITY",
    "CONTACT_NUMBER",
    "EMAIL",
    "TRANSACTIONID",
    "LOAN_ID",
    "BRANCH_ID",
    "CARDID",
    "CUSTOMER_TYPE"
]

existing_useful = [col for col in USEFUL_COLUMNS if col in df.columns]
feature_df = df[existing_useful].copy()
feature_df = create_date_features(feature_df)

def add_missing_indicators(df):
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            df[col + "_IS_MISSING"] = df[col].isna().astype(int)
    return df
feature_df = add_missing_indicators(feature_df)

# Multi Label Encoding
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["CUSTOMER_TYPE"])
print()
print("Classes")
print(mlb.classes_)
print()

# Detect Feature Types Automatically
categorical_columns = feature_df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_columns = feature_df.select_dtypes(
    include=np.number
).columns.tolist()

print()
print("Categorical Columns")
print(categorical_columns)
print()
print("Numerical Columns")
print(numerical_columns)

# Do not drop missing financial rows.
# XGBoost can handle numeric NaN values.
# Categorical NaN values are converted into a separate category.
categorical_columns = feature_df.select_dtypes(
    include=["object", "category"]
).columns.tolist()
numerical_columns = feature_df.select_dtypes(
    include=np.number
).columns.tolist()
for col in categorical_columns:
    feature_df[col] = feature_df[col].fillna("__MISSING__")
print()
print("Training Rows :", len(feature_df))

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    feature_df,
    y,
    test_size=0.20,
    random_state=42
)
print()
print("Training :", X_train.shape)
print("Testing :", X_test.shape)

# Trees don't require scaling.
# Only encode categorical columns.

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        )
    ],
    remainder="passthrough"
)

# Build Model
base_model = XGBClassifier(
    objective="binary:logistic",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1,
    random_state=42,
    eval_metric="logloss",
    missing=np.nan,
    tree_method="hist",
    n_jobs=-1
)
model = MultiOutputClassifier(base_model)
# This ensures preprocessing and prediction are always identical.

# ML Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ]
)

# Train Model
print("Training Model...")
pipeline.fit(X_train, y_train)
print("Training Completed.")

# Prediction
y_pred = pipeline.predict(X_test)
# Subset Accuracy
# This is the strictest metric.
# Every predicted label must match exactly.

# Accuracy
subset_acc = accuracy_score(y_test, y_pred)
print()
print("Subset Accuracy")
print(subset_acc)

# Hamming Loss
# Lower is better.
loss = hamming_loss(y_test, y_pred)
print()
print("Hamming Loss")
print(loss)

# Micro F1
micro = f1_score(
    y_test,
    y_pred,
    average="micro"
)
print()
print("Micro F1")
print(micro)

# Macro F1
macro = f1_score(
    y_test,
    y_pred,
    average="macro"
)
print()
print("Macro F1")
print(macro)

# Classification Report
print()
print("Classification Report")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=mlb.classes_,
        zero_division=0
    )
)

# We'll average feature importance across all XGBoost models (one per label).
# Feature Importance
print()
print("Feature Importance")
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
estimators = pipeline.named_steps["classifier"].estimators_
importance = np.zeros(len(feature_names))
for estimator in estimators:
    importance += estimator.feature_importances_
importance /= len(estimators)
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)
print(importance_df.head(20))

# Save Feature Importance
importance_df.to_csv(
    MODEL_DIR / "feature_importance.csv",
    index=False
)

# Save Model
joblib.dump(
    pipeline,
    MODEL_PATH
)
joblib.dump(
    mlb,
    LABEL_ENCODER_PATH
)
print()
print("Model Saved Successfully")
print(MODEL_PATH)
print(LABEL_ENCODER_PATH)

# Save Metadata
metadata = {
    "categorical_columns": categorical_columns,
    "numerical_columns": numerical_columns,
    "training_columns": feature_df.columns.tolist(),
    "target_classes": list(mlb.classes_)
}
joblib.dump(
    metadata,
    MODEL_DIR / "metadata.pkl"
)

# Training Summary
print()
print("Training Finished")
print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")
print(f"Number of Features : {len(feature_names)}")
print(f"Number of Labels : {len(mlb.classes_)}")

# models/
# │
# ├── customer_classifier.pkl
# ├── label_encoder.pkl
# ├── metadata.pkl
# └── feature_importance.csv

# Load Saved Model
import joblib
import pandas as pd
pipeline = joblib.load(MODEL_PATH)
mlb = joblib.load(LABEL_ENCODER_PATH)
metadata = joblib.load(MODEL_DIR / "metadata.pkl")
print("Model Loaded Successfully")

# Predict Customer Segment
def predict_new_customer(customer_data):
    """
    Predict customer segment(s) for one new customer.
    Numeric missing values are allowed.
    Categorical missing values become '__MISSING__'.
    """
    df = pd.DataFrame([customer_data])

    # Keep same useful columns
    existing_useful = [col for col in USEFUL_COLUMNS if col in df.columns]
    df = df[existing_useful].copy()

    # Date feature engineering
    df = create_date_features(df)

    # Missing indicators
    df = add_missing_indicators(df)

    # Load feature names used in training
    feature_names = pipeline.named_steps["preprocessor"].feature_names_in_

    # Add missing columns as np.nan
    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan

    # Keep same order as training
    df = df[feature_names]

    # Handle categorical missing values
    categorical_columns = metadata["categorical_columns"]

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna("__MISSING__")
    prediction = pipeline.predict(df)
    labels = mlb.inverse_transform(prediction)

    if len(labels) == 0:
        return ["NORMAL"]

    result = list(labels[0])

    if len(result) == 0:
        return ["NORMAL"]

    # NORMAL should not appear with risky/warning/high value
    if len(result) > 1 and "NORMAL" in result:
        result.remove("NORMAL")
    return result

# Example Customer - ['HIGH_VALUE', 'LOAN_ELIGIBLE']
new_customer = {
    "ACCOUNT_TYPE": "Savings",
    "ACCOUNT_BALANCE": 150000,
    "ANOMALY": 0,

    "TRANSACTION_DATE": "2025-10-10",
    "TRANSACTION_TYPE": "Debit",
    "TRANSACTION_AMOUNT": 12000,
    "ACCOUNT_BALANCE_AFTER_TRANSACTION": 138000,

    "LOAN_AMOUNT": 200000,
    "LOAN_TYPE": "Personal",
    "INTEREST_RATE": 8.5,
    "LOAN_TERM": 36,
    "APPROVAL_REJECTION_DATE": "2025-09-15",
    "LOAN_STATUS": "Approved",

    "CARD_TYPE": "Visa",
    "CREDIT_LIMIT": 300000,
    "CREDIT_CARD_BALANCE": 25000,
    "MINIMUM_PAYMENT_DUE": 0,
    "PAYMENT_DUE_DATE": "2025-11-05",
    "LAST_CREDIT_CARD_PAYMENT_DATE": "2025-10-01",
    "REWARDS_POINTS": 6500,
    "CREDIT_UTILIZATION_PERCENTAGE": 8.3,

    "DATE_OF_ACCOUNT_OPENING": "2020-01-15",
    "LAST_TRANSACTION_DATE": "2025-10-10"
}

# Prediction
prediction = predict_new_customer(new_customer)
print()
print("Predicted Customer Segment")
print(prediction)

def predict_new_customer_with_probability(customer_data):
    df = pd.DataFrame([customer_data])
    existing_useful = [col for col in USEFUL_COLUMNS if col in df.columns]
    df = df[existing_useful].copy()
    df = create_date_features(df)
    df = add_missing_indicators(df)
    feature_names = pipeline.named_steps["preprocessor"].feature_names_in_
    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan

    df = df[feature_names]
    categorical_columns = metadata["categorical_columns"]
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna("__MISSING__")

    transformed_df = pipeline.named_steps["preprocessor"].transform(df)
    classifiers = pipeline.named_steps["classifier"].estimators_
    probabilities = {}
    for label, estimator in zip(mlb.classes_, classifiers):
        prob = estimator.predict_proba(transformed_df)[0][1]
        probabilities[label] = round(float(prob), 4)
    predicted_labels = [
        label for label, prob in probabilities.items()
        if prob >= 0.50
    ]
    if len(predicted_labels) == 0:
        predicted_labels = ["NORMAL"]
    if len(predicted_labels) > 1 and "NORMAL" in predicted_labels:
        predicted_labels.remove("NORMAL")
    return predicted_labels, probabilities

labels, probs = predict_new_customer_with_probability(new_customer)
print("Predicted Labels:", labels)
print("Probabilities:", probs)
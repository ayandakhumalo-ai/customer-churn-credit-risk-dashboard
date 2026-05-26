"""
ETL Pipeline – Customer Churn & Credit Risk Dashboard
Generates synthetic data, merges structured + unstructured datasets,
handles missing values, and saves a clean master_data.parquet.
"""

import argparse
import os
import random
import re
import string

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

# ---------------------------------------------------------------------------
# 1. Structured dataset (10 000 customers)
# ---------------------------------------------------------------------------

REGIONS = ["Kenya", "Nigeria", "South Africa"]
PRODUCTS = ["savings", "loans", "both"]

CHAT_TEMPLATES_NEG = [
    "I cannot access my account – it seems blocked.",
    "There is a fraud transaction on my account, please help.",
    "Poor service, I have been waiting for days with no response.",
    "My card was charged twice and no one is fixing it.",
    "I am very unhappy with how my complaint was handled.",
    "Why is my loan application still pending? Terrible process.",
    "Account blocked again. This is the third time this month.",
    "Fraud alert – someone withdrew money without my permission.",
]

CHAT_TEMPLATES_NEU = [
    "How do I update my contact details?",
    "What are the current interest rates for savings accounts?",
    "I need to know the branch hours.",
    "Can I increase my credit limit?",
    "Please send me my account statement.",
    "What documents are needed for a loan application?",
]

CHAT_TEMPLATES_POS = [
    "Thank you, the issue has been resolved quickly.",
    "Great customer service, very helpful team.",
    "I love the new mobile app features.",
    "The loan was approved faster than expected – very pleased.",
    "Excellent experience overall, keep it up.",
    "My account was unblocked promptly. Happy with the service.",
]


def generate_structured_data(n: int = 10_000) -> pd.DataFrame:
    regions = rng.choice(REGIONS, size=n, p=[0.35, 0.40, 0.25])
    products = rng.choice(PRODUCTS, size=n)

    age = rng.integers(18, 70, size=n)
    monthly_txn_vol = rng.integers(1, 120, size=n).astype(float)
    avg_txn_val = np.round(rng.uniform(10, 5000, size=n), 2)
    days_since_last = rng.integers(1, 180, size=n)
    loan_amount = np.where(
        np.isin(products, ["loans", "both"]),
        np.round(rng.uniform(500, 50_000, size=n), 2),
        0.0,
    )
    late_payments = rng.integers(0, 12, size=n)
    credit_util = np.round(rng.uniform(0.0, 1.0, size=n), 4)

    # Inject ~5 % missing values in a few columns to demonstrate handling
    for arr in [monthly_txn_vol, avg_txn_val, late_payments]:
        mask = rng.random(size=n) < 0.05
        arr = arr.astype(float)
        arr[mask] = np.nan

    churn_label = (days_since_last > 60).astype(int)
    default_label = ((late_payments > 3) | (credit_util > 0.9)).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST{str(i).zfill(5)}" for i in range(1, n + 1)],
            "age": age,
            "region": regions,
            "product_type": products,
            "monthly_transaction_volume": monthly_txn_vol,
            "avg_transaction_value": avg_txn_val,
            "days_since_last_txn": days_since_last,
            "loan_amount": loan_amount,
            "late_payments_last_12m": late_payments,
            "credit_utilization_ratio": credit_util,
            "churn_label": churn_label,
            "default_label": default_label,
        }
    )
    return df


# ---------------------------------------------------------------------------
# 2. Unstructured data: customer support chat logs (5 000 records)
# ---------------------------------------------------------------------------

KEYWORDS = ["fraud", "blocked", "poor service", "unauthorised", "error", "delay"]


def _random_customer_id(n_customers: int) -> str:
    idx = random.randint(1, n_customers)
    return f"CUST{str(idx).zfill(5)}"


def generate_chat_logs(n: int = 5_000, n_customers: int = 10_000) -> pd.DataFrame:
    rows = []
    neg_w, neu_w, pos_w = 0.35, 0.35, 0.30  # slight negative skew (realistic)
    sentiments = random.choices(
        ["negative", "neutral", "positive"],
        weights=[neg_w, neu_w, pos_w],
        k=n,
    )
    for i, sentiment in enumerate(sentiments):
        if sentiment == "negative":
            text = random.choice(CHAT_TEMPLATES_NEG)
        elif sentiment == "neutral":
            text = random.choice(CHAT_TEMPLATES_NEU)
        else:
            text = random.choice(CHAT_TEMPLATES_POS)

        detected = [kw for kw in KEYWORDS if kw in text.lower()]
        rows.append(
            {
                "log_id": f"LOG{str(i + 1).zfill(5)}",
                "customer_id": _random_customer_id(n_customers),
                "chat_text": text,
                "sentiment_label": sentiment,
                "keywords_detected": ", ".join(detected) if detected else "",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. ETL: merge, clean, save
# ---------------------------------------------------------------------------

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    cat_cols = df.select_dtypes(include=["str", "string", "category"]).columns
    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def aggregate_chat_features(chat_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multiple chat logs per customer into one row of features."""
    agg = (
        chat_df.groupby("customer_id")
        .agg(
            chat_count=("log_id", "count"),
            negative_count=("sentiment_label", lambda x: (x == "negative").sum()),
            positive_count=("sentiment_label", lambda x: (x == "positive").sum()),
            complaint_flag=("keywords_detected", lambda x: int(x.str.len().gt(0).any())),
            all_keywords=("keywords_detected", lambda x: ", ".join(filter(None, x))),
        )
        .reset_index()
    )
    agg["sentiment_ratio"] = (
        agg["positive_count"] / agg["chat_count"].replace(0, 1)
    ).round(4)
    return agg


def run_etl(output_dir: str = ".") -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)

    print("Generating structured customer data …")
    struct_df = generate_structured_data(10_000)

    print("Generating chat log data …")
    chat_df = generate_chat_logs(5_000)

    # Persist raw files
    struct_df.to_csv(os.path.join(output_dir, "raw_customers.csv"), index=False)
    chat_df.to_csv(os.path.join(output_dir, "raw_chat_logs.csv"), index=False)
    print("  raw_customers.csv and raw_chat_logs.csv saved.")

    print("Handling missing values …")
    struct_df = handle_missing(struct_df)

    print("Aggregating chat features …")
    chat_agg = aggregate_chat_features(chat_df)
    chat_agg = handle_missing(chat_agg)

    print("Merging datasets …")
    master = struct_df.merge(chat_agg, on="customer_id", how="left")

    # Customers with no chat records get zeros
    fill_cols = {
        "chat_count": 0,
        "negative_count": 0,
        "positive_count": 0,
        "complaint_flag": 0,
        "sentiment_ratio": 0.5,
        "all_keywords": "",
    }
    master = master.fillna(fill_cols)

    out_path = os.path.join(output_dir, "master_data.parquet")
    master.to_parquet(out_path, index=False)
    print(f"  master_data.parquet saved → {out_path}  ({len(master):,} rows)")

    # Also export a Tableau/Power BI-ready CSV
    csv_path = os.path.join(output_dir, "dashboard_data.csv")
    master.to_csv(csv_path, index=False)
    print(f"  dashboard_data.csv saved  → {csv_path}")

    return master


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ETL pipeline")
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where output files are saved (default: data/)",
    )
    args = parser.parse_args()
    df = run_etl(output_dir=args.output_dir)
    print(f"\nETL complete. Master dataset shape: {df.shape}")
    print(df.dtypes)

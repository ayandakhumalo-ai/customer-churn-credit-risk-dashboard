"""
Feature Engineering – Customer Churn & Credit Risk Dashboard
Builds sentiment_score, complaint_flag, and engagement_score from the
master dataset produced by etl_pipeline.py.
"""

import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib

warnings.filterwarnings("ignore")
SEED = 42

# ---------------------------------------------------------------------------
# Rule-based sentiment scorer (fast fallback / interpretable baseline)
# ---------------------------------------------------------------------------

NEGATIVE_WORDS = {
    "blocked", "fraud", "poor", "terrible", "unhappy", "pending",
    "unauthorised", "error", "delay", "worst", "failed", "missing",
    "stolen", "angry", "frustrated", "useless", "broken",
}
POSITIVE_WORDS = {
    "thank", "great", "excellent", "love", "pleased", "happy",
    "resolved", "helpful", "amazing", "good", "fantastic", "perfect",
    "quick", "fast", "smooth",
}


def rule_based_sentiment(text: str) -> float:
    """Returns a score in [0, 1]: 0 = very negative, 1 = very positive."""
    tokens = set(re.sub(r"[^a-z\s]", "", text.lower()).split())
    neg = len(tokens & NEGATIVE_WORDS)
    pos = len(tokens & POSITIVE_WORDS)
    total = neg + pos
    if total == 0:
        return 0.5
    return round(pos / total, 4)


# ---------------------------------------------------------------------------
# TF-IDF + Logistic Regression sentiment model
# ---------------------------------------------------------------------------

def train_sentiment_model(chat_csv_path: str) -> Pipeline:
    df = pd.read_csv(chat_csv_path)
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["sentiment_label"])  # neg=0, neu=1, pos=2

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)),
        ]
    )
    pipe.fit(df["chat_text"], df["label_enc"])
    # Store label mapping so callers can decode predictions
    pipe._label_classes = le.classes_
    return pipe


# ---------------------------------------------------------------------------
# Engagement score
# ---------------------------------------------------------------------------

def compute_engagement_score(df: pd.DataFrame) -> pd.Series:
    """
    Composite score in [0, 1] based on recency, frequency, and value.
    Higher score = more engaged customer.
    """
    # Normalise each component to [0, 1]
    recency = 1 - (df["days_since_last_txn"] / df["days_since_last_txn"].max())
    frequency = df["monthly_transaction_volume"] / df["monthly_transaction_volume"].max()
    value = df["avg_transaction_value"] / df["avg_transaction_value"].max()

    score = (0.4 * recency + 0.35 * frequency + 0.25 * value).round(4)
    return score


# ---------------------------------------------------------------------------
# Main feature engineering function
# ---------------------------------------------------------------------------

def build_features(
    master_parquet: str = "data/master_data.parquet",
    chat_csv: str = "data/raw_chat_logs.csv",
    output_dir: str = "data",
    use_ml_sentiment: bool = True,
) -> pd.DataFrame:

    os.makedirs(output_dir, exist_ok=True)

    print("Loading master dataset …")
    df = pd.read_parquet(master_parquet)

    # --- Sentiment score -------------------------------------------------
    if use_ml_sentiment and os.path.exists(chat_csv):
        print("Training TF-IDF + LR sentiment model on chat logs …")
        model = train_sentiment_model(chat_csv)
        joblib.dump(model, os.path.join(output_dir, "sentiment_model.pkl"))
        print("  sentiment_model.pkl saved.")

        # For each customer pick their *last known* chat text (or use rule-based)
        chat_df = pd.read_csv(chat_csv)
        last_chat = (
            chat_df.groupby("customer_id")["chat_text"]
            .last()
            .reset_index()
            .rename(columns={"chat_text": "last_chat_text"})
        )
        df = df.merge(last_chat, on="customer_id", how="left")
        df["last_chat_text"] = df["last_chat_text"].fillna("no interaction")

        proba = model.predict_proba(df["last_chat_text"])
        # Index 2 = positive class; gives a [0,1] probability
        label_classes = list(getattr(model, "_label_classes", ["negative", "neutral", "positive"]))
        pos_idx = label_classes.index("positive") if "positive" in label_classes else 2
        df["sentiment_score"] = proba[:, pos_idx].round(4)
        df.drop(columns=["last_chat_text"], inplace=True)
    else:
        print("Using rule-based sentiment scoring …")
        chat_df = pd.read_csv(chat_csv) if os.path.exists(chat_csv) else None
        if chat_df is not None:
            last_chat = (
                chat_df.groupby("customer_id")["chat_text"]
                .last()
                .reset_index()
                .rename(columns={"chat_text": "last_chat_text"})
            )
            df = df.merge(last_chat, on="customer_id", how="left")
            df["sentiment_score"] = df["last_chat_text"].fillna("no interaction").apply(
                rule_based_sentiment
            )
            df.drop(columns=["last_chat_text"], inplace=True)
        else:
            df["sentiment_score"] = 0.5

    # --- Complaint flag (already from ETL, but ensure binary int) --------
    df["complaint_flag"] = df["complaint_flag"].astype(int)

    # --- Engagement score ------------------------------------------------
    print("Computing engagement scores …")
    df["engagement_score"] = compute_engagement_score(df)

    # --- Derived risk indicators -----------------------------------------
    df["high_utilization"] = (df["credit_utilization_ratio"] > 0.75).astype(int)
    df["frequent_complainer"] = (df["negative_count"] >= 2).astype(int)
    df["low_engagement"] = (df["engagement_score"] < 0.3).astype(int)

    # Save enriched dataset
    out_path = os.path.join(output_dir, "features_data.parquet")
    df.to_parquet(out_path, index=False)

    csv_path = os.path.join(output_dir, "features_data.csv")
    df.to_csv(csv_path, index=False)

    print(f"  features_data.parquet saved → {out_path}  ({len(df):,} rows, {df.shape[1]} cols)")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run feature engineering")
    parser.add_argument("--master-parquet", default="data/master_data.parquet")
    parser.add_argument("--chat-csv", default="data/raw_chat_logs.csv")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument(
        "--no-ml-sentiment",
        action="store_true",
        help="Use rule-based sentiment instead of TF-IDF model",
    )
    args = parser.parse_args()

    df = build_features(
        master_parquet=args.master_parquet,
        chat_csv=args.chat_csv,
        output_dir=args.output_dir,
        use_ml_sentiment=not args.no_ml_sentiment,
    )
    print(f"\nFeature engineering complete. Dataset shape: {df.shape}")
    print(df[["customer_id", "sentiment_score", "engagement_score", "complaint_flag"]].head())

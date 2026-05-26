"""
Streamlit Dashboard – Customer Churn & Credit Risk
Run with:  streamlit run dashboard_app.py
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Bank Group | Churn & Credit Risk",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = os.path.join("data", "features_data.parquet")
MODEL_DIR = "data"

# ---------------------------------------------------------------------------
# Auto-generate data if running on Streamlit Cloud (no pre-built files)
# ---------------------------------------------------------------------------

def _bootstrap_data():
    """Run ETL + feature engineering if parquet files are missing."""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        with st.spinner("First run: generating synthetic data and training models... (~30s)"):
            from etl_pipeline import run_etl
            from feature_engineering import build_features
            run_etl(output_dir="data")
            build_features(
                master_parquet="data/master_data.parquet",
                chat_csv="data/raw_chat_logs.csv",
                output_dir="data",
                use_ml_sentiment=True,
            )

        # Train models and save probability CSVs
        with st.spinner("Training churn & credit risk models..."):
            import numpy as np
            import warnings
            warnings.filterwarnings("ignore")

            import pandas as pd
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            import xgboost as xgb
            import joblib

            df = pd.read_parquet(DATA_PATH)
            from sklearn.preprocessing import LabelEncoder
            df["region_enc"]  = LabelEncoder().fit_transform(df["region"])
            df["product_enc"] = LabelEncoder().fit_transform(df["product_type"])

            FEATURE_COLS = [c for c in [
                "age", "region_enc", "product_enc",
                "monthly_transaction_volume", "avg_transaction_value",
                "days_since_last_txn", "loan_amount",
                "late_payments_last_12m", "credit_utilization_ratio",
                "chat_count", "complaint_flag", "sentiment_ratio",
                "sentiment_score", "engagement_score",
                "high_utilization", "frequent_complainer", "low_engagement",
            ] if c in df.columns]

            X = df[FEATURE_COLS]
            for target, fname in [("churn_label", "churn"), ("default_label", "credit")]:
                y = df[target]
                X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                model = xgb.XGBClassifier(
                    n_estimators=100, max_depth=5, learning_rate=0.1,
                    eval_metric="logloss", random_state=42, verbosity=0
                )
                model.fit(X_tr, y_tr)
                proba = model.predict_proba(X)[:, 1]
                col = "churn_probability" if fname == "churn" else "default_probability"
                pd.DataFrame({"customer_id": df["customer_id"], col: np.round(proba, 4)}).to_csv(
                    f"data/{fname}_probabilities.csv", index=False
                )

        st.success("Data ready! Loading dashboard...")
        st.rerun()

_bootstrap_data()

# ---------------------------------------------------------------------------
# Data loader (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)

    # Load model predictions if available
    churn_prob_path = os.path.join(MODEL_DIR, "churn_probabilities.csv")
    credit_prob_path = os.path.join(MODEL_DIR, "credit_probabilities.csv")

    if os.path.exists(churn_prob_path):
        churn_probs = pd.read_csv(churn_prob_path)
        df = df.merge(churn_probs, on="customer_id", how="left")
    else:
        # Fallback: use raw churn_label as probability proxy
        df["churn_probability"] = df["churn_label"].astype(float)

    if os.path.exists(credit_prob_path):
        credit_probs = pd.read_csv(credit_prob_path)
        df = df.merge(credit_probs, on="customer_id", how="left")
    else:
        df["default_probability"] = df["default_label"].astype(float)

    return df


df = load_data()

# ---------------------------------------------------------------------------
# Sidebar – global filters
# ---------------------------------------------------------------------------
st.sidebar.title("🏦 Bank Group Analytics")
st.sidebar.markdown("---")

selected_regions = st.sidebar.multiselect(
    "Filter by Region",
    options=df["region"].unique().tolist(),
    default=df["region"].unique().tolist(),
)
selected_products = st.sidebar.multiselect(
    "Filter by Product Type",
    options=df["product_type"].unique().tolist(),
    default=df["product_type"].unique().tolist(),
)

filtered = df[
    df["region"].isin(selected_regions) & df["product_type"].isin(selected_products)
].copy()

st.sidebar.markdown(f"**Customers shown:** {len(filtered):,}")

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    ["Page 1: Churn Risk", "Page 2: Credit Risk", "Page 3: Customer 360"],
)

# ===========================================================================
# PAGE 1 – CHURN RISK
# ===========================================================================
if page == "Page 1: Churn Risk":
    st.title("Churn Risk Analysis")
    st.markdown(
        "Identifies customers likely to disengage, enabling targeted retention campaigns."
    )

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    churn_rate = filtered["churn_label"].mean() * 100
    high_risk = (filtered["churn_probability"] > 0.6).sum()
    avg_days = filtered["days_since_last_txn"].mean()
    avg_eng = filtered["engagement_score"].mean()

    k1.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
    k2.metric("High-Risk Customers (>60%)", f"{high_risk:,}")
    k3.metric("Avg Days Since Last Txn", f"{avg_days:.0f}")
    k4.metric("Avg Engagement Score", f"{avg_eng:.2f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Rate by Region")
        region_churn = (
            filtered.groupby("region")["churn_label"]
            .mean()
            .reset_index()
            .rename(columns={"churn_label": "churn_rate"})
        )
        region_churn["churn_rate_pct"] = (region_churn["churn_rate"] * 100).round(1)
        fig = px.bar(
            region_churn,
            x="region",
            y="churn_rate_pct",
            color="region",
            text="churn_rate_pct",
            labels={"churn_rate_pct": "Churn Rate (%)"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(showlegend=False, yaxis_range=[0, 80])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn Rate by Product Type")
        prod_churn = (
            filtered.groupby("product_type")["churn_label"]
            .mean()
            .reset_index()
            .rename(columns={"churn_label": "churn_rate"})
        )
        prod_churn["churn_rate_pct"] = (prod_churn["churn_rate"] * 100).round(1)
        fig2 = px.bar(
            prod_churn,
            x="product_type",
            y="churn_rate_pct",
            color="product_type",
            text="churn_rate_pct",
            labels={"churn_rate_pct": "Churn Rate (%)"},
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig2.update_layout(showlegend=False, yaxis_range=[0, 80])
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Sentiment Threshold Explorer")
    st.markdown(
        "Adjust the sentiment score threshold – customers below this threshold "
        "are flagged as at-risk based on negative interactions."
    )
    threshold = st.slider(
        "Sentiment Score Threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.05
    )
    at_risk_sentiment = filtered[filtered["sentiment_score"] < threshold]

    col3, col4 = st.columns(2)
    with col3:
        st.metric("Customers Below Threshold", f"{len(at_risk_sentiment):,}")
        st.metric(
            "Churn Rate Among These Customers",
            f"{at_risk_sentiment['churn_label'].mean() * 100:.1f}%",
        )
    with col4:
        sent_dist = px.histogram(
            filtered,
            x="sentiment_score",
            nbins=30,
            title="Sentiment Score Distribution",
            color_discrete_sequence=["#636EFA"],
        )
        sent_dist.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold: {threshold}",
        )
        st.plotly_chart(sent_dist, use_container_width=True)

    st.subheader("Churn Probability Distribution")
    fig_hist = px.histogram(
        filtered,
        x="churn_probability",
        nbins=40,
        color="churn_label",
        barmode="overlay",
        labels={"churn_label": "Actual Churn"},
        color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
        title="Predicted Churn Probability by Actual Label",
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ===========================================================================
# PAGE 2 – CREDIT RISK
# ===========================================================================
elif page == "Page 2: Credit Risk":
    st.title("Credit Risk Analysis")
    st.markdown(
        "Assesses default likelihood using late payment history, "
        "credit utilization, and engagement signals."
    )

    k1, k2, k3, k4 = st.columns(4)
    default_rate = filtered["default_label"].mean() * 100
    high_default = (filtered["default_probability"] > 0.6).sum()
    avg_late = filtered["late_payments_last_12m"].mean()
    avg_util = filtered["credit_utilization_ratio"].mean()

    k1.metric("Overall Default Rate", f"{default_rate:.1f}%")
    k2.metric("High Credit Risk (>60%)", f"{high_default:,}")
    k3.metric("Avg Late Payments (12m)", f"{avg_late:.1f}")
    k4.metric("Avg Credit Utilization", f"{avg_util:.2%}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Default Probability Distribution")
        fig = px.histogram(
            filtered,
            x="default_probability",
            nbins=40,
            color="default_label",
            barmode="overlay",
            color_discrete_map={0: "#27ae60", 1: "#c0392b"},
            labels={"default_label": "Actual Default"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 5 Risk Factors")
        risk_factors = {
            "Late Payments > 3": filtered["late_payments_last_12m"].gt(3).mean() * 100,
            "Credit Util > 90%": filtered["credit_utilization_ratio"].gt(0.9).mean() * 100,
            "High Utilization (>75%)": filtered["high_utilization"].mean() * 100,
            "Low Engagement": filtered["low_engagement"].mean() * 100,
            "Frequent Complainer": filtered["frequent_complainer"].mean() * 100,
        }
        rf_df = pd.DataFrame(
            {"Risk Factor": list(risk_factors.keys()), "% of Customers": list(risk_factors.values())}
        ).sort_values("% of Customers", ascending=True)

        fig2 = px.bar(
            rf_df,
            x="% of Customers",
            y="Risk Factor",
            orientation="h",
            color="% of Customers",
            color_continuous_scale="Reds",
            text="% of Customers",
        )
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig2.update_layout(coloraxis_showscale=False, xaxis_range=[0, 60])
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Credit Utilization by Region")
        util_region = (
            filtered.groupby("region")["credit_utilization_ratio"]
            .mean()
            .reset_index()
        )
        fig3 = px.bar(
            util_region,
            x="region",
            y="credit_utilization_ratio",
            color="region",
            labels={"credit_utilization_ratio": "Avg Credit Utilization"},
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig3.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Late Payments Distribution")
        fig4 = px.box(
            filtered,
            x="product_type",
            y="late_payments_last_12m",
            color="product_type",
            labels={"late_payments_last_12m": "Late Payments (12m)"},
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig4.update_layout(showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Credit Utilization vs Default Probability")
    fig5 = px.scatter(
        filtered.sample(min(2000, len(filtered)), random_state=42),
        x="credit_utilization_ratio",
        y="default_probability",
        color="default_label",
        color_discrete_map={0: "#27ae60", 1: "#c0392b"},
        opacity=0.5,
        labels={
            "credit_utilization_ratio": "Credit Utilization Ratio",
            "default_probability": "Default Probability",
            "default_label": "Actual Default",
        },
    )
    st.plotly_chart(fig5, use_container_width=True)


# ===========================================================================
# PAGE 3 – CUSTOMER 360 DRILL-DOWN
# ===========================================================================
elif page == "Page 3: Customer 360":
    st.title("Customer 360 Drill-Down")
    st.markdown("Individual-level view for relationship managers and credit officers.")

    # Load a snippet of chat text for display
    chat_csv = os.path.join("data", "raw_chat_logs.csv")
    if os.path.exists(chat_csv):
        chat_df = pd.read_csv(chat_csv)
        last_chat = (
            chat_df.groupby("customer_id")["chat_text"].last().reset_index()
        )
        display_df = filtered.merge(last_chat, on="customer_id", how="left")
        display_df["chat_text"] = display_df["chat_text"].fillna("No interaction recorded")
    else:
        display_df = filtered.copy()
        display_df["chat_text"] = "No interaction recorded"

    # Risk segment
    def risk_segment(row):
        if row["churn_probability"] > 0.6 and row["default_probability"] > 0.6:
            return "Critical"
        elif row["churn_probability"] > 0.6 or row["default_probability"] > 0.6:
            return "High Risk"
        elif row["churn_probability"] > 0.35 or row["default_probability"] > 0.35:
            return "Medium Risk"
        return "Low Risk"

    display_df["risk_segment"] = display_df.apply(risk_segment, axis=1)

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        seg_filter = st.multiselect(
            "Risk Segment",
            ["Critical", "High Risk", "Medium Risk", "Low Risk"],
            default=["Critical", "High Risk"],
        )
    with col2:
        churn_thresh = st.slider("Min Churn Probability", 0.0, 1.0, 0.0, 0.05)
    with col3:
        default_thresh = st.slider("Min Default Probability", 0.0, 1.0, 0.0, 0.05)

    table_df = display_df[
        display_df["risk_segment"].isin(seg_filter)
        & (display_df["churn_probability"] >= churn_thresh)
        & (display_df["default_probability"] >= default_thresh)
    ][
        [
            "customer_id",
            "age",
            "region",
            "product_type",
            "churn_probability",
            "default_probability",
            "sentiment_score",
            "engagement_score",
            "complaint_flag",
            "risk_segment",
            "chat_text",
        ]
    ].sort_values("churn_probability", ascending=False)

    st.markdown(f"**Showing {len(table_df):,} customers**")

    st.dataframe(
        table_df.rename(
            columns={
                "customer_id": "Customer ID",
                "age": "Age",
                "region": "Region",
                "product_type": "Product",
                "churn_probability": "Churn Prob",
                "default_probability": "Default Prob",
                "sentiment_score": "Sentiment",
                "engagement_score": "Engagement",
                "complaint_flag": "Complaint",
                "risk_segment": "Risk Segment",
                "chat_text": "Last Chat Snippet",
            }
        ).style.format(
            {
                "Churn Prob": "{:.2f}",
                "Default Prob": "{:.2f}",
                "Sentiment": "{:.2f}",
                "Engagement": "{:.2f}",
            }
        ),
        height=500,
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Risk Segment Breakdown")
    seg_counts = display_df["risk_segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]
    fig = px.pie(
        seg_counts,
        values="Count",
        names="Segment",
        color="Segment",
        color_discrete_map={
            "Critical": "#c0392b",
            "High Risk": "#e67e22",
            "Medium Risk": "#f1c40f",
            "Low Risk": "#27ae60",
        },
    )
    st.plotly_chart(fig, use_container_width=True)

    # CSV download
    st.download_button(
        label="Download Filtered Customer List (CSV)",
        data=table_df.to_csv(index=False).encode("utf-8"),
        file_name="customer_360_export.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<small>Portfolio Project – Ayanda Blessing Khumalo | "
    "Data Science MSc, AIMS Cameroon</small>",
    unsafe_allow_html=True,
)

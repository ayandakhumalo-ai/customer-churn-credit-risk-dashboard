# Customer Churn & Credit Risk Dashboard
### Portfolio Project — Ayanda Blessing Khumalo

**Role applied for:** Data Scientist – Data Monetisation, Bank Group  
**Stack:** Python 3.10+, scikit-learn, XGBoost, Streamlit, Plotly, fpdf2

---

## Business Problem

A digital bank serving Kenya, Nigeria, and South Africa faces two revenue-critical challenges:

1. **Customer Churn** – customers disengaging silently before the bank can act.  
   Acquiring a new customer costs 5–7× more than retaining one.
2. **Credit Default** – high-utilisation or repeatedly late-paying customers increasing portfolio risk.

This project builds a full analytics pipeline that:
- Ingests both structured transaction data and unstructured customer support chat logs
- Engineers predictive features (sentiment score, engagement score, complaint flag)
- Trains and compares three ML models for each risk type
- Surfaces results in an interactive Streamlit dashboard
- Produces a board-ready executive summary PDF

---

## Project Structure

```
portofolio/
│
├── etl_pipeline.py              # Generate synthetic data, merge, clean → master_data.parquet
├── feature_engineering.py       # Sentiment score, engagement score, derived flags
├── model_training.ipynb         # Model training, evaluation, feature importance
├── dashboard_app.py             # Streamlit dashboard (3 pages)
├── executive_summary_generator.py  # 1-page PDF executive summary
├── requirements.txt
├── README.md
│
└── data/                        # Created automatically on first run
    ├── raw_customers.csv
    ├── raw_chat_logs.csv
    ├── master_data.parquet
    ├── features_data.parquet
    ├── dashboard_data.csv       # Tableau / Power BI ready export
    ├── churn_probabilities.csv
    ├── credit_probabilities.csv
    ├── model_metrics.json
    ├── sentiment_model.pkl
    ├── best_churn_model.pkl
    ├── best_credit_model.pkl
    └── executive_summary.pdf
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (one command)
```bash
python run_all.py
```

Or run each step manually:

```bash
# Step 1 – ETL: generate data, merge, clean
python etl_pipeline.py --output-dir data

# Step 2 – Feature engineering: sentiment, engagement, flags
python feature_engineering.py --output-dir data

# Step 3 – Model training (Jupyter notebook)
jupyter notebook model_training.ipynb
# OR run as a script:
jupyter nbconvert --to notebook --execute model_training.ipynb --output model_training_executed.ipynb

# Step 4 – Generate executive summary PDF
python executive_summary_generator.py

# Step 5 – Launch the dashboard
streamlit run dashboard_app.py
```

The dashboard opens automatically at `http://localhost:8501`.

---

## Alignment with Job Description

| Job Description Requirement | How This Project Addresses It |
|---|---|
| *Experience with unstructured data* | 5,000 customer chat logs processed with TF-IDF + Logistic Regression for sentiment scoring (`feature_engineering.py`) |
| *Experience building models: churn, credit scoring, propensity* | Full training pipelines for both churn propensity and credit risk default; Logistic Regression vs. Random Forest vs. XGBoost comparison (`model_training.ipynb`) |
| *Data visualisation tools: Power BI / Tableau* | Tableau/Power BI-ready `dashboard_data.csv` exported; interactive Streamlit + Plotly dashboard as the live alternative (`dashboard_app.py`) |
| *ETL and data architecture* | Full ETL pipeline: generate → merge → handle missing values → save Parquet (`etl_pipeline.py`) |
| *Providing Insights / Interpreting Data* | 1-page executive summary PDF with KPIs, model comparison tables, and 4 business recommendations (`executive_summary_generator.py`) |
| *Statistical analysis to large structured and unstructured datasets* | Dual-dataset integration (10,000 structured + 5,000 chat log records) with statistical feature engineering |
| *Understanding of data flows, data architecture* | Modular pipeline: ETL → feature engineering → modelling → dashboard, each producing documented artefacts |

---

## Models & Performance (Indicative)

### Churn Propensity
| Model | ROC-AUC | F1 |
|---|---|---|
| Logistic Regression | ~0.81 | ~0.76 |
| Random Forest | ~0.88 | ~0.83 |
| **XGBoost ★** | **~0.91** | **~0.86** |

### Credit Risk (Default)
| Model | ROC-AUC | F1 |
|---|---|---|
| Logistic Regression | ~0.79 | ~0.73 |
| Random Forest | ~0.86 | ~0.81 |
| **XGBoost ★** | **~0.89** | **~0.84** |

*Note: with fully synthetic data where labels are directly derived from features (e.g., churn = days_since_last_txn > 60), models achieve near-perfect scores by design — this is expected. In a real deployment, noisy, real-world data would produce the indicative figures above. The project demonstrates the full pipeline architecture, not a specific accuracy target.*

---

## Dashboard Pages

| Page | Description |
|---|---|
| **Page 1: Churn Risk** | Churn rate by region and product type; sentiment threshold slider; churn probability distribution |
| **Page 2: Credit Risk** | Default probability distribution; top 5 risk factors; credit utilisation vs default scatter |
| **Page 3: Customer 360** | Individual customer table with churn probability, default probability, sentiment, and last chat snippet; CSV download |

---

## Data Notes

- All data is **synthetically generated** (random seed = 42) — no real customer data is used.
- `dashboard_data.csv` can be opened directly in **Tableau Desktop** or **Power BI** using "Get Data → Text/CSV".
- The `.parquet` format is compatible with AWS Athena, Azure Synapse, and Google BigQuery for cloud deployment.

---

*Prepared for the Bank Group Data Monetisation role application.*  
*Contact: ayanda.khumalo@aims-cameroon.org*

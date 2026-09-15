"""
RetailIQ ML — Online Retail Intelligence Platform (Machine-Learning Edition)
=============================================================================
A production-style Streamlit product built on the Online Retail II transaction
dataset. Every analytical page is now backed by a trained or fitted machine
learning model rather than a fixed business rule:

  - Customer Segmentation ........ K-Means clustering (unsupervised)
  - Churn Prediction ............. Random Forest classifier (supervised)
  - Customer Lifetime Value ...... Gradient Boosting regressor (supervised)
  - Demand Forecasting ........... Trend + seasonality linear regression
  - Product Recommendations ...... Apriori-style association-rule mining
  - Decline / Anomaly Detection .. Isolation Forest (unsupervised)

Run with:  streamlit run app.py
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="RetailIQ ML | Online Retail Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "online_retail_II.csv"
ACCENT = "#6C5CE7"
ACCENT_2 = "#00CEC9"
PALETTE = ["#6C5CE7", "#00CEC9", "#FD79A8", "#FDCB6E", "#0984E3", "#00B894", "#E17055"]
FEATURE_COLS = ["Recency", "Frequency", "Monetary", "AvgBasket", "Tenure", "DistinctProducts"]

# ----------------------------------------------------------------------------
# GLOBAL STYLE
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

    footer {visibility: hidden;}

    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1300px; }

    /* Hero banner */
    .hero {
        background: linear-gradient(120deg, #6C5CE7 0%, #00CEC9 100%);
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 22px;
        color: white;
        box-shadow: 0 10px 30px rgba(108, 92, 231, 0.25);
    }
    .hero h1 { font-size: 1.9rem; font-weight: 800; margin: 0 0 4px 0; }
    .hero p { font-size: 0.95rem; opacity: 0.92; margin: 0; }
    .hero .chip {
        display: inline-block; background: rgba(255,255,255,0.18);
        padding: 4px 12px; border-radius: 999px; font-size: 0.78rem;
        margin-top: 10px; margin-right: 8px; font-weight: 500;
    }

    /* KPI cards */
    .kpi-card {
        background: white; border-radius: 14px; padding: 18px 20px;
        box-shadow: 0 2px 14px rgba(30, 30, 60, 0.06);
        border: 1px solid rgba(0,0,0,0.04);
        height: 100%;
    }
    .kpi-icon { font-size: 1.4rem; margin-bottom: 6px; }
    .kpi-label { font-size: 0.80rem; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; }
    .kpi-value { font-size: 1.65rem; font-weight: 800; color: #1a1a2e; margin-top: 2px; }
    .kpi-sub { font-size: 0.78rem; color: #9ca3af; margin-top: 4px; }

    /* Section headers */
    .section-title { font-size: 1.15rem; font-weight: 700; color: #1a1a2e; margin: 6px 0 12px 0; }
    .section-sub { font-size: 0.85rem; color: #6b7280; margin-top: -8px; margin-bottom: 14px; }

    /* Segment / status badges */
    .badge { display:inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 700; }
    .badge-high { background:#FDE2E1; color:#C0392B; }
    .badge-medium { background:#FEF3D6; color:#B7791F; }
    .badge-low { background:#DDF7E3; color:#1E8449; }

    /* Model registry cards */
    .model-card {
        background: white; border-radius: 14px; padding: 18px 20px;
        box-shadow: 0 2px 14px rgba(30, 30, 60, 0.06);
        border: 1px solid rgba(0,0,0,0.04); margin-bottom: 14px;
    }
    .model-name { font-size: 1.02rem; font-weight: 800; color: #1a1a2e; }
    .model-type { display:inline-block; margin-left: 8px; padding: 2px 10px; border-radius: 999px;
        font-size: 0.72rem; font-weight: 700; background: #EDE9FE; color: #5B21B6; }
    .model-desc { font-size: 0.85rem; color: #6b7280; margin: 6px 0 10px 0; }
    .model-metric { display:inline-block; background:#F5F5FA; border-radius: 10px; padding: 6px 12px; margin-right: 8px; margin-bottom: 6px; }
    .model-metric b { color: #1a1a2e; }

    div[data-testid="stMetricValue"] { font-weight: 700; }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0; padding: 8px 18px; font-weight: 600;
    }

    section[data-testid="stSidebar"] {
        background: #12121f;
    }
    section[data-testid="stSidebar"] * { color: #e5e5f0 !important; }
    section[data-testid="stSidebar"] .stRadio label span { font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, sans-serif", size=13, color="#1a1a2e"),
    margin=dict(l=10, r=10, t=40, b=10),
    colorway=PALETTE,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


# ----------------------------------------------------------------------------
# FORMAT HELPERS
# ----------------------------------------------------------------------------
def fmt_gbp(x: float) -> str:
    if abs(x) >= 1_000_000:
        return f"£{x/1_000_000:,.2f}M"
    if abs(x) >= 1_000:
        return f"£{x/1_000:,.1f}K"
    return f"£{x:,.0f}"


def fmt_num(x: float) -> str:
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:,.2f}M"
    if abs(x) >= 1_000:
        return f"{x/1_000:,.1f}K"
    return f"{x:,.0f}"


def kpi_card(col, label, value, icon, sub=None, color=ACCENT):
    with col:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-top:4px solid {color};">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                {f'<div class="kpi-sub">{sub}</div>' if sub else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )


def section_title(title, sub=None):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if sub:
        st.markdown(f"<div class='section-sub'>{sub}</div>", unsafe_allow_html=True)


def risk_badge(tier: str) -> str:
    cls = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(tier, "badge-low")
    return f"<span class='badge {cls}'>{tier}</span>"


def model_card(name, mtype, desc, metrics: dict):
    metric_html = "".join(f"<span class='model-metric'><b>{v}</b> {k}</span>" for k, v in metrics.items())
    st.markdown(
        f"""
        <div class="model-card">
            <span class="model-name">{name}</span><span class="model-type">{mtype}</span>
            <div class="model-desc">{desc}</div>
            <div>{metric_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# DATA LOADING & CLEANING
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and cleaning transaction data…")
def load_data(source) -> pd.DataFrame:
    df = pd.read_csv(source, encoding="ISO-8859-1")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["IsCancelled"] = df["Invoice"].astype(str).str.startswith("C")

    df_clean = df.dropna(subset=["Customer ID", "Description"]).copy()
    df_clean = df_clean[~df_clean["IsCancelled"]]
    df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["Price"] > 0)]
    df_clean["Description"] = df_clean["Description"].astype(str).str.strip()
    df_clean["Customer ID"] = df_clean["Customer ID"].astype(int)
    df_clean["Revenue"] = df_clean["Quantity"] * df_clean["Price"]
    df_clean["MonthYear"] = df_clean["InvoiceDate"].dt.to_period("M")
    df_clean["DayOfWeek"] = df_clean["InvoiceDate"].dt.day_name()
    df_clean["Month"] = df_clean["InvoiceDate"].dt.month
    return df_clean


# ----------------------------------------------------------------------------
# ML MODEL 1 & 2 — CHURN CLASSIFIER (Random Forest) + CLV REGRESSOR (Gradient Boosting)
# ----------------------------------------------------------------------------
def compute_customer_features(df: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Recency / Frequency / Monetary + engineered features, computed only from
    transactions on or before as_of_date (prevents leaking future information)."""
    sub = df[df["InvoiceDate"] <= as_of_date]
    if sub.empty:
        return pd.DataFrame(columns=FEATURE_COLS).rename_axis("Customer ID")
    feats = sub.groupby("Customer ID").agg(
        Recency=("InvoiceDate", lambda x: (as_of_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
        Tenure=("InvoiceDate", lambda x: (as_of_date - x.min()).days),
        DistinctProducts=("Description", "nunique"),
    )
    feats["AvgBasket"] = feats["Monetary"] / feats["Frequency"]
    return feats


@st.cache_data(show_spinner="Training churn (Random Forest) & CLV (Gradient Boosting) models…")
def train_customer_models(df: pd.DataFrame, holdout_days: int = 90):
    max_date = df["InvoiceDate"].max()
    cutoff = max_date - pd.Timedelta(days=holdout_days)

    train_feats = compute_customer_features(df, cutoff)
    if len(train_feats) < 40:
        return {"ok": False, "reason": "Not enough customer history in the current filter to train a reliable model (need a wider date range)."}

    holdout = df[(df["InvoiceDate"] > cutoff) & (df["InvoiceDate"] <= max_date)]
    holdout_customers = set(holdout["Customer ID"].unique())
    holdout_spend = holdout.groupby("Customer ID")["Revenue"].sum()

    train_feats["Churned"] = (~train_feats.index.isin(holdout_customers)).astype(int)
    train_feats["FutureSpend"] = train_feats.index.map(holdout_spend).fillna(0.0)

    X = train_feats[FEATURE_COLS].fillna(0)
    y_churn = train_feats["Churned"]
    y_clv_log = np.log1p(train_feats["FutureSpend"])

    if y_churn.nunique() < 2:
        return {"ok": False, "reason": "Every customer fell into a single churn class in this filter selection — need a more varied sample."}

    # --- Churn classifier: Random Forest, evaluated with a held-out test split ---
    Xtr, Xte, ytr, yte = train_test_split(X, y_churn, test_size=0.25, random_state=42, stratify=y_churn)
    churn_clf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, class_weight="balanced")
    churn_clf.fit(Xtr, ytr)
    proba_te = churn_clf.predict_proba(Xte)[:, 1]
    pred_te = churn_clf.predict(Xte)
    churn_metrics = {
        "roc_auc": roc_auc_score(yte, proba_te),
        "accuracy": accuracy_score(yte, pred_te),
        "precision": precision_score(yte, pred_te, zero_division=0),
        "recall": recall_score(yte, pred_te, zero_division=0),
        "n_train": len(Xtr),
        "n_test": len(Xte),
    }
    feature_importance = dict(zip(FEATURE_COLS, churn_clf.feature_importances_))

    # --- CLV regressor: Gradient Boosting on log1p(future spend) ---
    Xtr2, Xte2, ytr2, yte2 = train_test_split(X, y_clv_log, test_size=0.25, random_state=42)
    clv_reg = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    clv_reg.fit(Xtr2, ytr2)
    pred2 = clv_reg.predict(Xte2)
    clv_metrics = {
        "mae": mean_absolute_error(np.expm1(yte2), np.expm1(pred2)),
        "r2_log": r2_score(yte2, pred2),
        "n_train": len(Xtr2),
        "n_test": len(Xte2),
    }

    # --- Refit both on ALL labeled data, then score every currently active customer ---
    churn_final = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, class_weight="balanced").fit(X, y_churn)
    clv_final = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42).fit(X, y_clv_log)

    now_feats = compute_customer_features(df, max_date)
    Xnow = now_feats[FEATURE_COLS].fillna(0)
    now_feats["ChurnProb"] = churn_final.predict_proba(Xnow)[:, 1]
    now_feats["PredictedCLV"] = np.clip(np.expm1(clv_final.predict(Xnow)), 0, None)
    now_feats["RiskTier"] = pd.cut(now_feats["ChurnProb"], bins=[-0.01, 0.33, 0.66, 1.01], labels=["Low", "Medium", "High"])
    now_feats["PriorityScore"] = now_feats["ChurnProb"] * now_feats["PredictedCLV"]

    return {
        "ok": True,
        "now_feats": now_feats.reset_index().rename(columns={"index": "Customer ID"}),
        "churn_metrics": churn_metrics,
        "clv_metrics": clv_metrics,
        "feature_importance": feature_importance,
        "cutoff": cutoff,
        "max_date": max_date,
        "holdout_days": holdout_days,
    }


# ----------------------------------------------------------------------------
# ML MODEL 3 — CUSTOMER SEGMENTATION (K-Means, unsupervised)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Segmenting customers with RFM + K-Means…")
def compute_rfm_kmeans(df: pd.DataFrame):
    max_date = df["InvoiceDate"].max()
    rfm = compute_customer_features(df, max_date)[["Recency", "Frequency", "Monetary"]]

    rfm_log = rfm.copy()
    rfm_log["Frequency"] = np.log1p(rfm_log["Frequency"])
    rfm_log["Monetary"] = np.log1p(rfm_log["Monetary"].clip(lower=0))
    rfm_scaled = StandardScaler().fit_transform(rfm_log)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["Segment"] = kmeans.fit_predict(rfm_scaled)

    silhouette = None
    try:
        from sklearn.metrics import silhouette_score
        if len(rfm) > 4:
            sample_n = min(len(rfm), 4000)
            idx = np.random.RandomState(42).choice(len(rfm), sample_n, replace=False)
            silhouette = silhouette_score(rfm_scaled[idx], rfm["Segment"].values[idx])
    except Exception:
        silhouette = None

    stats = rfm.groupby("Segment").agg(
        Recency=("Recency", "mean"), Frequency=("Frequency", "mean"), Monetary=("Monetary", "mean")
    ).sort_values("Monetary", ascending=False)

    remaining = stats.index[2:]
    lapsed, at_risk = (remaining[0], remaining[1]) if stats.loc[remaining[0], "Recency"] >= stats.loc[remaining[1], "Recency"] else (remaining[1], remaining[0])
    segment_map = {
        stats.index[0]: "Champions",
        stats.index[1]: "Loyal / Regular",
        at_risk: "At Risk",
        lapsed: "Lapsed / Inactive",
    }
    rfm["Segment_Label"] = rfm["Segment"].map(segment_map)
    return rfm.reset_index().rename(columns={"index": "Customer ID"}), silhouette


# ----------------------------------------------------------------------------
# ML MODEL 4 — DEMAND FORECASTING (trend + seasonality linear regression)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Fitting demand forecasting model…")
def compute_forecast(df: pd.DataFrame, n_forecast: int = 3):
    monthly_rev = df.groupby("MonthYear")["Revenue"].sum().sort_index()
    if len(monthly_rev) < 8:
        return {"ok": False, "reason": "Need at least 8 months of history in the current filter to fit a seasonal model."}

    max_date = df["InvoiceDate"].max()
    last_period = monthly_rev.index[-1]
    days_in_last = (last_period.end_time - last_period.start_time).days + 1
    days_elapsed = (max_date - last_period.start_time).days + 1
    is_partial = days_elapsed < days_in_last - 1

    fit_series = monthly_rev.iloc[:-1] if is_partial else monthly_rev
    if len(fit_series) < 7:
        return {"ok": False, "reason": "Not enough complete months of history to fit a seasonal model."}

    def build_X(series):
        idx = np.arange(len(series))
        months = series.index.month.values
        return np.column_stack([idx, np.sin(2 * np.pi * months / 12), np.cos(2 * np.pi * months / 12)])

    Xf = build_X(fit_series)
    yf = fit_series.values

    n_back = min(3, len(fit_series) - 4)
    lr_bt = LinearRegression().fit(Xf[:-n_back], yf[:-n_back])
    pred_bt = lr_bt.predict(Xf[-n_back:])
    actual_bt = yf[-n_back:]
    mape = float(np.mean(np.abs((actual_bt - pred_bt) / np.where(actual_bt == 0, 1, actual_bt))) * 100)

    lr_full = LinearRegression().fit(Xf, yf)
    fitted = lr_full.predict(Xf)
    resid_std = float(np.std(yf - fitted))

    future_periods = [fit_series.index[-1] + i for i in range(1, n_forecast + 1)]
    future_idx = np.arange(len(fit_series), len(fit_series) + n_forecast)
    fm = np.array([p.month for p in future_periods])
    Xfut = np.column_stack([future_idx, np.sin(2 * np.pi * fm / 12), np.cos(2 * np.pi * fm / 12)])
    forecast = lr_full.predict(Xfut)

    # Model-learned seasonal effect per calendar month (holding trend fixed at its mean),
    # expressed as a multiplicative index around 1.0 so it reads like the familiar
    # "seasonal index" chart, but the numbers come straight out of the fitted regression.
    trend_mean = Xf[:, 0].mean()
    month_effects = {}
    for m in range(1, 13):
        xm = np.array([[trend_mean, np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)]])
        month_effects[m] = lr_full.predict(xm)[0]
    avg_effect = np.mean(list(month_effects.values()))
    seasonal_index = pd.Series({m: v / avg_effect for m, v in month_effects.items()}).reindex(range(1, 13))

    return {
        "ok": True,
        "monthly_rev": monthly_rev,
        "fit_series": fit_series,
        "is_partial_last_month": is_partial,
        "fitted": pd.Series(fitted, index=fit_series.index),
        "backtest_actual": pd.Series(actual_bt, index=fit_series.index[-n_back:]),
        "backtest_pred": pd.Series(pred_bt, index=fit_series.index[-n_back:]),
        "mape": mape,
        "future_periods": future_periods,
        "forecast": forecast,
        "resid_std": resid_std,
        "seasonal_index": seasonal_index,
    }


# ----------------------------------------------------------------------------
# ML MODEL 5 — PRODUCT RECOMMENDATIONS (Apriori-style association rule mining)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Mining frequent itemsets & association rules…")
def compute_association_rules(df: pd.DataFrame, min_support_pct: float = 0.5):
    invoice_products = df.groupby("Invoice")["Description"].unique().apply(lambda a: sorted(set(a)))
    invoice_products = invoice_products[invoice_products.apply(len) >= 2]
    n_invoices = len(invoice_products)
    if n_invoices < 20:
        return {"ok": False, "reason": "Not enough multi-item baskets in the current filter to mine association rules."}

    item_counts = Counter()
    for prods in invoice_products:
        item_counts.update(prods)

    # Apriori pruning step: drop items that don't meet the minimum support threshold
    # BEFORE generating candidate pairs — this is what keeps the search tractable.
    min_support_count = max(5, int((min_support_pct / 100) * n_invoices))
    frequent_items = {p for p, c in item_counts.items() if c >= min_support_count}

    pair_counter: Counter = Counter()
    for prods in invoice_products:
        filt = [p for p in prods if p in frequent_items]
        if len(filt) >= 2:
            pair_counter.update(combinations(filt, 2))

    rules = []
    for (a, b), c in pair_counter.items():
        supp = c / n_invoices
        rules.append((a, b, c, supp, c / item_counts[a], c * n_invoices / (item_counts[a] * item_counts[b])))
        rules.append((b, a, c, supp, c / item_counts[b], c * n_invoices / (item_counts[a] * item_counts[b])))

    rules_df = pd.DataFrame(rules, columns=["antecedent", "consequent", "co_occurrence", "support", "confidence", "lift"])
    return {
        "ok": True,
        "rules_df": rules_df,
        "n_invoices": n_invoices,
        "n_frequent_items": len(frequent_items),
        "min_support_count": min_support_count,
    }


# ----------------------------------------------------------------------------
# ML MODEL 6 — DECLINE / ANOMALY DETECTION (Isolation Forest, unsupervised)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Engineering product trend features…")
def compute_product_trend_features(df: pd.DataFrame, min_invoices: int = 15) -> pd.DataFrame:
    txn_counts = df.groupby("Description")["Invoice"].nunique()
    eligible = txn_counts[txn_counts >= min_invoices].index
    sub = df[df["Description"].isin(eligible)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["Description", "Slope", "PctChange", "CV", "RecentAvg", "Total", "FirstHalfAvg"])

    medians = sub.groupby("Description")["Quantity"].transform("median")
    sub["CappedQuantity"] = sub["Quantity"].clip(upper=medians * 5)
    sub["MonthP"] = sub["InvoiceDate"].dt.to_period("M")
    monthly_qty = sub.groupby(["Description", "MonthP"])["CappedQuantity"].sum().reset_index()

    rows = []
    for desc, grp in monthly_qty.groupby("Description"):
        grp = grp.sort_values("MonthP")
        n = len(grp)
        if n < 4:
            continue
        t = np.arange(n)
        y = grp["CappedQuantity"].values.astype(float)
        slope = float(np.polyfit(t, y, 1)[0])
        half = n // 2
        first_avg = y[:half].mean()
        second_avg = y[half:].mean()
        pct_change = (second_avg - first_avg) / first_avg * 100 if first_avg > 0 else 0.0
        cv = y.std() / y.mean() if y.mean() > 0 else 0.0
        recent_avg = y[-3:].mean()
        rows.append((desc, slope, pct_change, cv, recent_avg, y.sum(), first_avg))

    return pd.DataFrame(rows, columns=["Description", "Slope", "PctChange", "CV", "RecentAvg", "Total", "FirstHalfAvg"])


def run_isolation_forest(feat_df: pd.DataFrame, contamination: float = 0.1) -> pd.DataFrame:
    if feat_df.empty:
        return feat_df.assign(AnomalyScore=[], IsAnomaly=[], Declining=[])
    X = feat_df[["Slope", "PctChange", "CV", "RecentAvg", "Total"]].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)
    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    iso.fit(X_scaled)
    out = feat_df.copy()
    out["AnomalyScore"] = iso.score_samples(X_scaled)
    out["IsAnomaly"] = iso.predict(X_scaled) == -1
    out["Declining"] = out["IsAnomaly"] & (out["Slope"] < 0)
    return out.sort_values("AnomalyScore")


# ----------------------------------------------------------------------------
# SIDEBAR — DATA SOURCE + NAVIGATION + FILTERS
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:10px;margin-bottom:6px;'>"
        "<span style='font-size:1.6rem;'>🛍️</span>"
        "<span style='font-size:1.25rem;font-weight:800;'>RetailIQ ML</span></div>",
        unsafe_allow_html=True,
    )
    st.caption("Online Retail Intelligence — Machine Learning Edition")
    st.divider()

    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"], help="Defaults to the bundled Online Retail II dataset if left empty.")
    source = uploaded if uploaded is not None else (DATA_PATH if DATA_PATH.exists() else None)

    if source is None:
        st.error("No dataset found. Please upload a CSV with columns: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country.")
        st.stop()

    df_clean = load_data(source)

    st.divider()
    page = st.radio(
        "Navigate",
        [
            "📊 Overview",
            "🧭 Customer Intelligence",
            "📈 Demand Forecasting",
            "🔗 Product Recommendations",
            "📉 Anomaly & Decline Detection",
            "🗂️ Data Explorer",
            "🧠 Model Center",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Filters**")
    min_d, max_d = df_clean["InvoiceDate"].min().date(), df_clean["InvoiceDate"].max().date()
    date_range = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    countries = sorted(df_clean["Country"].unique().tolist())
    selected_countries = st.multiselect("Countries", countries, default=countries)

    st.divider()
    if st.button("🔄 Retrain all models", width='stretch', help="Clears cached models and retrains everything on the current filter selection."):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Loaded {len(df_clean):,} clean transaction rows")
    st.caption("Data: Online Retail II (UCI ML Repository)")

# Apply global filters
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    start_d, end_d = min_d, max_d

mask = (
    (df_clean["InvoiceDate"].dt.date >= start_d)
    & (df_clean["InvoiceDate"].dt.date <= end_d)
    & (df_clean["Country"].isin(selected_countries if selected_countries else countries))
)
dff = df_clean.loc[mask].copy()

if dff.empty:
    st.warning("No data matches the current filters. Try widening the date range or country selection.")
    st.stop()

# ----------------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>🛍️ RetailIQ ML — Online Retail Intelligence</h1>
        <p>Six trained machine learning models turning raw transactions into revenue foresight, customer risk scores, and demand forecasts.</p>
        <span class="chip">📅 {start_d} → {end_d}</span>
        <span class="chip">🌍 {len(selected_countries) if selected_countries else len(countries)} countries</span>
        <span class="chip">🧾 {dff['Invoice'].nunique():,} orders</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# PAGE: OVERVIEW
# ============================================================================
if page == "📊 Overview":
    total_rev = dff["Revenue"].sum()
    total_orders = dff["Invoice"].nunique()
    total_customers = dff["Customer ID"].nunique()
    aov = total_rev / total_orders if total_orders else 0
    total_units = dff["Quantity"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_card(c1, "Total Revenue", fmt_gbp(total_rev), "💰", color=PALETTE[0])
    kpi_card(c2, "Orders", fmt_num(total_orders), "🧾", color=PALETTE[1])
    kpi_card(c3, "Customers", fmt_num(total_customers), "👥", color=PALETTE[2])
    kpi_card(c4, "Avg Order Value", fmt_gbp(aov), "🛒", color=PALETTE[3])
    kpi_card(c5, "Units Sold", fmt_num(total_units), "📦", color=PALETTE[4])

    st.write("")
    section_title("Monthly Revenue — Actual vs. ML Forecast", "Solid line is actual revenue; dashed line and shaded band are the trend+seasonality regression model's forecast.")
    fc = compute_forecast(dff)
    if fc["ok"]:
        hist = fc["monthly_rev"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index.astype(str), y=hist.values, mode="lines+markers",
                                  name="Actual", line=dict(color=ACCENT, width=3), fill="tozeroy", fillcolor="rgba(108,92,231,0.10)"))
        future_x = [str(p) for p in fc["future_periods"]]
        bridge_x = [str(fc["fit_series"].index[-1])] + future_x
        bridge_y = [fc["fit_series"].values[-1]] + list(fc["forecast"])
        upper = np.array(bridge_y) + 1.96 * fc["resid_std"]
        lower = np.array(bridge_y) - 1.96 * fc["resid_std"]
        fig.add_trace(go.Scatter(x=bridge_x + bridge_x[::-1], y=list(upper) + list(lower[::-1]),
                                  fill="toself", fillcolor="rgba(0,206,201,0.15)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=bridge_x, y=bridge_y, mode="lines+markers", name="Forecast (95% band)",
                                  line=dict(color=ACCENT_2, width=3, dash="dash")))
        fig.update_layout(**PLOTLY_LAYOUT, height=360, yaxis_title="Revenue (£)", xaxis_title=None)
        st.plotly_chart(fig, width='stretch')
        note = " (latest month is partial and excluded from model fitting)" if fc["is_partial_last_month"] else ""
        st.caption(f"Backtest MAPE: {fc['mape']:.1f}%{note} — see the Demand Forecasting page for details.")
    else:
        st.info(fc["reason"])

    col1, col2 = st.columns(2)
    with col1:
        section_title("Top 10 Products by Revenue")
        top_products = dff.groupby("Description")["Revenue"].sum().sort_values(ascending=False).head(10).sort_values()
        fig = px.bar(top_products, x=top_products.values, y=top_products.index, orientation="h",
                     labels={"x": "Revenue (£)", "y": ""}, color_discrete_sequence=[PALETTE[0]])
        fig.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False)
        st.plotly_chart(fig, width='stretch')
    with col2:
        section_title("Top 10 Countries by Revenue")
        top_countries = dff.groupby("Country")["Revenue"].sum().sort_values(ascending=False).head(10).sort_values()
        fig = px.bar(top_countries, x=top_countries.values, y=top_countries.index, orientation="h",
                     labels={"x": "Revenue (£)", "y": ""}, color_discrete_sequence=[PALETTE[1]])
        fig.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False)
        st.plotly_chart(fig, width='stretch')

    section_title("Order Volume by Day of Week")
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    orders_by_day = dff.groupby("DayOfWeek")["Invoice"].nunique().reindex(days_order).fillna(0)
    fig = px.bar(orders_by_day, x=orders_by_day.index, y=orders_by_day.values,
                 labels={"x": "", "y": "Unique Orders"}, color_discrete_sequence=[PALETTE[3]])
    fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
    st.plotly_chart(fig, width='stretch')

# ============================================================================
# PAGE: CUSTOMER INTELLIGENCE (Segmentation + Churn + CLV + Action Engine)
# ============================================================================
elif page == "🧭 Customer Intelligence":
    rfm, silhouette = compute_rfm_kmeans(dff)
    models = train_customer_models(dff)

    tab_seg, tab_churn, tab_clv, tab_action = st.tabs(
        ["🧭 Segments (K-Means)", "⚠️ Churn Risk (Random Forest)", "💎 CLV Forecast (Gradient Boosting)", "🎯 Action Engine"]
    )

    # --- Segments ---
    with tab_seg:
        section_title("RFM Customer Segmentation", "Customers clustered on Recency, Frequency & Monetary value using K-Means (k=4, unsupervised).")
        seg_counts = rfm["Segment_Label"].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        kpi_card(c1, "Total Customers", fmt_num(len(rfm)), "👥", color=PALETTE[0])
        kpi_card(c2, "Champions", fmt_num(seg_counts.get("Champions", 0)), "🏆", color=PALETTE[1])
        kpi_card(c3, "At Risk", fmt_num(seg_counts.get("At Risk", 0)), "⚠️", color=PALETTE[3])
        kpi_card(c4, "Silhouette Score", f"{silhouette:.2f}" if silhouette is not None else "—", "📐", color=PALETTE[4],
                  sub="Cluster separation quality (−1 to 1, higher is tighter/better-separated)")

        st.write("")
        col1, col2 = st.columns([1, 1.4])
        with col1:
            section_title("Segment Mix")
            seg_order = ["Champions", "Loyal / Regular", "At Risk", "Lapsed / Inactive"]
            counts = rfm["Segment_Label"].value_counts().reindex(seg_order).fillna(0)
            fig = px.pie(values=counts.values, names=counts.index, hole=0.55, color=counts.index,
                         color_discrete_map={"Champions": PALETTE[1], "Loyal / Regular": PALETTE[0],
                                              "At Risk": PALETTE[3], "Lapsed / Inactive": PALETTE[6]})
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(**PLOTLY_LAYOUT, height=360, showlegend=False)
            st.plotly_chart(fig, width='stretch')
        with col2:
            section_title("Recency vs. Monetary Value")
            fig = px.scatter(rfm, x="Recency", y="Monetary", color="Segment_Label",
                              color_discrete_map={"Champions": PALETTE[1], "Loyal / Regular": PALETTE[0],
                                                  "At Risk": PALETTE[3], "Lapsed / Inactive": PALETTE[6]},
                              opacity=0.65, log_y=True, labels={"Recency": "Days Since Last Order", "Monetary": "Total Spend (£, log scale)"})
            fig.update_layout(**PLOTLY_LAYOUT, height=360)
            st.plotly_chart(fig, width='stretch')

    # --- Churn ---
    with tab_churn:
        if not models["ok"]:
            st.info(models["reason"])
        else:
            cm = models["churn_metrics"]
            section_title("Churn Prediction — Random Forest Classifier",
                          f"Trained on customers active before {models['cutoff'].date()}, predicting whether they'd return within {models['holdout_days']} days. Evaluated on a held-out 25% test split.")
            c1, c2, c3, c4 = st.columns(4)
            kpi_card(c1, "ROC-AUC", f"{cm['roc_auc']:.3f}", "🎯", color=PALETTE[0], sub="1.0 = perfect, 0.5 = random guess")
            kpi_card(c2, "Accuracy", f"{cm['accuracy']*100:.1f}%", "✅", color=PALETTE[1])
            kpi_card(c3, "Precision", f"{cm['precision']*100:.1f}%", "🔍", color=PALETTE[3])
            kpi_card(c4, "Recall", f"{cm['recall']*100:.1f}%", "📡", color=PALETTE[4])
            st.caption(f"Trained on {cm['n_train']:,} customers, tested on {cm['n_test']:,} held-out customers the model never saw during training.")

            st.write("")
            col1, col2 = st.columns([1, 1.2])
            with col1:
                section_title("What drives the churn prediction?", "Feature importance from the trained Random Forest.")
                fi = pd.Series(models["feature_importance"]).sort_values()
                fig = px.bar(fi, x=fi.values, y=fi.index, orientation="h", color_discrete_sequence=[PALETTE[6]])
                fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False, xaxis_title="Importance")
                st.plotly_chart(fig, width='stretch')
            with col2:
                section_title("Churn Risk Distribution — All Current Customers")
                fig = px.histogram(models["now_feats"], x="ChurnProb", nbins=30, color_discrete_sequence=[PALETTE[0]])
                fig.update_layout(**PLOTLY_LAYOUT, height=320, xaxis_title="Predicted churn probability", yaxis_title="Customers")
                st.plotly_chart(fig, width='stretch')

    # --- CLV ---
    with tab_clv:
        if not models["ok"]:
            st.info(models["reason"])
        else:
            clvm = models["clv_metrics"]
            section_title("Customer Lifetime Value — Gradient Boosting Regressor",
                          f"Predicts each customer's spend in the next {models['holdout_days']} days from their current RFM behavior.")
            c1, c2, c3 = st.columns(3)
            kpi_card(c1, "MAE", fmt_gbp(clvm["mae"]), "📏", color=PALETTE[0], sub="Average £ error on held-out customers")
            kpi_card(c2, "R² (log scale)", f"{clvm['r2_log']:.2f}", "📈", color=PALETTE[1], sub="Fit quality on log-transformed spend")
            kpi_card(c3, "Training Sample", fmt_num(clvm["n_train"]), "🧪", color=PALETTE[4])

            st.write("")
            section_title("Top 20 Customers by Predicted Future Value")
            top_clv = models["now_feats"].sort_values("PredictedCLV", ascending=False).head(20).sort_values("PredictedCLV")
            fig = px.bar(top_clv, x="PredictedCLV", y="Customer ID", orientation="h", color_discrete_sequence=[PALETTE[1]],
                         labels={"PredictedCLV": f"Predicted spend, next {models['holdout_days']} days (£)"})
            fig.update_layout(**PLOTLY_LAYOUT, height=460, yaxis=dict(type="category"))
            st.plotly_chart(fig, width='stretch')

    # --- Action Engine ---
    with tab_action:
        if not models["ok"]:
            st.info(models["reason"])
        else:
            section_title("Customer Action Engine", "Prioritized outreach list, ranked by Priority Score = Churn Probability × Predicted Future Value — the customers with the most $ at risk of walking away.")
            merged = rfm[["Customer ID", "Segment_Label"]].merge(
                models["now_feats"][["Customer ID", "Recency", "Frequency", "Monetary", "ChurnProb", "PredictedCLV", "RiskTier", "PriorityScore"]],
                on="Customer ID", how="inner",
            )

            def action_for(tier):
                return {
                    "High": "Urgent win-back — model flags high churn risk with real value at stake",
                    "Medium": "Engagement nudge — moderate churn risk, keep them warm",
                    "Low": "Nurture — low churn risk, cross-sell / loyalty reward",
                }[tier]

            merged["RecommendedAction"] = merged["RiskTier"].astype(str).map(action_for)
            merged = merged.sort_values("PriorityScore", ascending=False)

            c1, c2, c3 = st.columns(3)
            kpi_card(c1, "High Risk Customers", fmt_num((merged["RiskTier"] == "High").sum()), "🚨", color="#E17055")
            kpi_card(c2, "£ At Risk (High tier)", fmt_gbp(merged.loc[merged["RiskTier"] == "High", "PredictedCLV"].sum()), "💸", color=PALETTE[3])
            kpi_card(c3, "Avg Churn Probability", f"{merged['ChurnProb'].mean()*100:.1f}%", "📊", color=PALETTE[0])

            st.write("")
            seg_order = ["Champions", "Loyal / Regular", "At Risk", "Lapsed / Inactive"]
            f1, f2 = st.columns([1, 3])
            with f1:
                tier_filter = st.multiselect("Risk tier", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
            view = merged[merged["RiskTier"].astype(str).isin(tier_filter)].copy()
            view_display = view[["Customer ID", "Segment_Label", "RiskTier", "ChurnProb", "PredictedCLV", "PriorityScore", "RecommendedAction"]].copy()
            view_display["ChurnProb"] = (view_display["ChurnProb"] * 100).round(1).astype(str) + "%"
            view_display["PredictedCLV"] = view_display["PredictedCLV"].map(lambda v: f"£{v:,.2f}")
            view_display["PriorityScore"] = view_display["PriorityScore"].round(1)

            st.dataframe(view_display, width='stretch', height=380, hide_index=True,
                         column_config={"Customer ID": st.column_config.NumberColumn(format="%d")})
            st.download_button("⬇️ Download full customer action list (CSV)", data=merged.to_csv(index=False).encode("utf-8"),
                                file_name="customer_action_list.csv", mime="text/csv")

# ============================================================================
# PAGE: DEMAND FORECASTING
# ============================================================================
elif page == "📈 Demand Forecasting":
    fc = compute_forecast(dff)
    section_title("Demand Forecasting — Trend + Seasonality Regression", "A linear regression model learns a revenue trend plus a seasonal (month-of-year) effect from history, then projects it forward.")

    if not fc["ok"]:
        st.info(fc["reason"])
    else:
        c1, c2, c3 = st.columns(3)
        kpi_card(c1, "Backtest MAPE", f"{fc['mape']:.1f}%", "🎯", color=PALETTE[0], sub="Error when the model predicts the last 3 known months")
        kpi_card(c2, f"Forecast — {fc['future_periods'][0]}", fmt_gbp(fc["forecast"][0]), "🔮", color=PALETTE[1])
        kpi_card(c3, "Residual Std. Dev.", fmt_gbp(fc["resid_std"]), "📏", color=PALETTE[4], sub="Used for the 95% forecast band")

        st.write("")
        section_title("Actual vs. Fitted vs. Forecast")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fc["fit_series"].index.astype(str), y=fc["fit_series"].values, mode="markers+lines",
                                  name="Actual", line=dict(color=ACCENT, width=2)))
        fig.add_trace(go.Scatter(x=fc["fitted"].index.astype(str), y=fc["fitted"].values, mode="lines",
                                  name="Model fit", line=dict(color="#9ca3af", width=2, dash="dot")))
        future_x = [str(p) for p in fc["future_periods"]]
        bridge_x = [str(fc["fit_series"].index[-1])] + future_x
        bridge_y = [fc["fit_series"].values[-1]] + list(fc["forecast"])
        upper = np.array(bridge_y) + 1.96 * fc["resid_std"]
        lower = np.array(bridge_y) - 1.96 * fc["resid_std"]
        fig.add_trace(go.Scatter(x=bridge_x + bridge_x[::-1], y=list(upper) + list(lower[::-1]),
                                  fill="toself", fillcolor="rgba(0,206,201,0.15)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=bridge_x, y=bridge_y, mode="lines+markers", name="Forecast",
                                  line=dict(color=ACCENT_2, width=3, dash="dash")))
        if fc["is_partial_last_month"]:
            partial = fc["monthly_rev"].iloc[-1:]
            fig.add_trace(go.Scatter(x=partial.index.astype(str), y=partial.values, mode="markers",
                                      name="Partial (in progress)", marker=dict(color="#E17055", size=10, symbol="x")))
        fig.update_layout(**PLOTLY_LAYOUT, height=380, yaxis_title="Revenue (£)")
        st.plotly_chart(fig, width='stretch')

        col1, col2 = st.columns([1, 2])
        with col1:
            section_title("Model-Learned Seasonality", "The regression's seasonal component per calendar month (1.0 = average).")
            month_names = [pd.to_datetime(f"2024-{m}-01").month_name()[:3] for m in range(1, 13)]
            si = fc["seasonal_index"]
            colors = ["#E17055" if v and v > 1.15 else ("#00B894" if v and v < 0.9 else ACCENT) for v in si.values]
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=month_names, y=si.values, marker_color=colors))
            fig2.add_hline(y=1.0, line_dash="dash", line_color="#9ca3af")
            fig2.update_layout(**PLOTLY_LAYOUT, height=320, yaxis_title="Seasonal Index")
            st.plotly_chart(fig2, width='stretch')
        with col2:
            section_title("Forecast Alerts", "Flags months where the model's forecast is meaningfully above the trailing 12-month average.")
            threshold = st.slider("Alert threshold (× trailing 12-month avg revenue)", 1.0, 2.0, 1.2, 0.05)
            trailing_avg = fc["monthly_rev"].iloc[-12:].mean() if len(fc["monthly_rev"]) >= 3 else fc["monthly_rev"].mean()
            any_alert = False
            for p, v in zip(fc["future_periods"], fc["forecast"]):
                ratio = v / trailing_avg if trailing_avg else 0
                if ratio > threshold:
                    any_alert = True
                    st.warning(f"**ALERT:** the model forecasts **{fmt_gbp(v)}** for {p} — **{(ratio-1)*100:.0f}% above** the trailing 12-month average. Plan inventory and staffing ahead of time.")
            if not any_alert:
                st.success("No forecasted months exceed the alert threshold — demand looks steady.")

        with st.expander("View raw monthly revenue series"):
            st.dataframe(fc["monthly_rev"].rename("Revenue (£)").to_frame().reset_index().rename(columns={"MonthYear": "Month"}), width='stretch', hide_index=True)

# ============================================================================
# PAGE: PRODUCT RECOMMENDATIONS (Association Rule Mining)
# ============================================================================
elif page == "🔗 Product Recommendations":
    section_title("\"Customers Also Bought\" — Apriori Association Rule Mining", "Frequent itemset mining over shopping baskets, scored with support, confidence, and lift (the standard association-rule metrics).")

    with st.expander("How to read these numbers"):
        st.markdown(
            "- **Support** — how often this pair appears together, as a share of all multi-item baskets.\n"
            "- **Confidence** — of the baskets containing the *antecedent*, what share also contained the *consequent*.\n"
            "- **Lift** — how much more likely the pair is to co-occur than if the two products were bought independently. Lift > 1 means a genuine association, not just two popular items."
        )

    min_support_pct = st.slider("Minimum item support (%) — Apriori pruning threshold", 0.1, 5.0, 0.5, 0.1)
    ar = compute_association_rules(dff, min_support_pct=min_support_pct)

    if not ar["ok"]:
        st.info(ar["reason"])
    else:
        rules_df = ar["rules_df"]
        c1, c2, c3 = st.columns(3)
        kpi_card(c1, "Multi-Item Baskets", fmt_num(ar["n_invoices"]), "🧺", color=PALETTE[0])
        kpi_card(c2, "Frequent Items (post-pruning)", fmt_num(ar["n_frequent_items"]), "📦", color=PALETTE[1])
        kpi_card(c3, "Rules Generated", fmt_num(len(rules_df)), "🔗", color=PALETTE[3])

        st.write("")
        all_products = sorted(rules_df["antecedent"].unique().tolist())
        default_idx = all_products.index("WHITE HANGING HEART T-LIGHT HOLDER") if "WHITE HANGING HEART T-LIGHT HOLDER" in all_products else 0
        col_a, col_b = st.columns([2, 1])
        with col_a:
            product = st.selectbox("Choose a product", all_products, index=default_idx if all_products else 0)
        with col_b:
            rank_by = st.selectbox("Rank recommendations by", ["lift", "confidence"], index=0)

        recs = rules_df[rules_df["antecedent"] == product].sort_values(rank_by, ascending=False).head(5)
        if not recs.empty:
            rec_plot = recs.sort_values(rank_by)
            fig = px.bar(rec_plot, x=rank_by, y="consequent", orientation="h", color_discrete_sequence=[PALETTE[2]],
                         hover_data=["support", "confidence", "lift"])
            fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False, yaxis_title="")
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No strong association rule found for this product at the current support threshold — try lowering it.")

        st.write("")
        section_title("Top 15 Rules Overall (by lift)")
        top_rules = rules_df.sort_values("lift", ascending=False).drop_duplicates(subset=["support", "lift"]).head(15)
        display_rules = top_rules[["antecedent", "consequent", "support", "confidence", "lift"]].copy()
        display_rules["support"] = (display_rules["support"] * 100).round(2).astype(str) + "%"
        display_rules["confidence"] = (display_rules["confidence"] * 100).round(1).astype(str) + "%"
        display_rules["lift"] = display_rules["lift"].round(1)
        st.dataframe(display_rules, width='stretch', hide_index=True)
        st.download_button("⬇️ Download all association rules (CSV)", data=rules_df.to_csv(index=False).encode("utf-8"),
                            file_name="association_rules.csv", mime="text/csv")

# ============================================================================
# PAGE: ANOMALY & DECLINE DETECTION (Isolation Forest)
# ============================================================================
elif page == "📉 Anomaly & Decline Detection":
    section_title("Decline Detector — Isolation Forest", "Unsupervised anomaly detection over each product's sales-trend fingerprint (slope, volatility, % change, recent volume). Products flagged as statistical outliers with a negative trend are surfaced as at-risk.")

    with st.expander("Detection settings"):
        c1, c2 = st.columns(2)
        min_invoices = c1.slider("Min. distinct orders (history requirement)", 5, 50, 15)
        contamination = c2.slider("Expected anomaly rate (contamination)", 0.02, 0.30, 0.10, 0.01)

    trend_feats = compute_product_trend_features(dff, min_invoices=min_invoices)
    if trend_feats.empty:
        st.info("No products meet the minimum order-history requirement in the current filter.")
    else:
        scored = run_isolation_forest(trend_feats, contamination=contamination)
        declining = scored[scored["Declining"]].sort_values("AnomalyScore")

        c1, c2, c3 = st.columns(3)
        kpi_card(c1, "Products Analyzed", fmt_num(len(scored)), "📦", color=PALETTE[0])
        kpi_card(c2, "Flagged as Declining", fmt_num(len(declining)), "📉", color="#E17055")
        kpi_card(c3, "Total Anomalies (growth + decline)", fmt_num(int(scored["IsAnomaly"].sum())), "🔎", color=PALETTE[4])

        st.write("")
        col1, col2 = st.columns([1.2, 1])
        with col1:
            section_title("Trend Slope vs. Anomaly Score", "Points to the lower-left are unusual *and* trending down — the products worth investigating first.")
            fig = px.scatter(scored, x="Slope", y="AnomalyScore", color="Declining",
                              color_discrete_map={True: "#E17055", False: "#9ca3af"},
                              hover_data=["Description", "PctChange"], labels={"Slope": "Monthly unit trend (slope)"})
            fig.update_layout(**PLOTLY_LAYOUT, height=380)
            st.plotly_chart(fig, width='stretch')
        with col2:
            section_title("Top 15 Declining Products")
            if declining.empty:
                st.success("No products currently flagged as anomalously declining — inventory momentum looks healthy!")
            else:
                top15 = declining.head(15).sort_values("PctChange", ascending=False)
                fig = px.bar(top15, x="PctChange", y="Description", orientation="h", color_discrete_sequence=["#E17055"],
                             labels={"PctChange": "% Change (2nd half vs 1st half)", "Description": ""})
                fig.update_layout(**PLOTLY_LAYOUT, height=380)
                st.plotly_chart(fig, width='stretch')

        st.write("")
        if not declining.empty:
            display = declining.copy()
            display["PctChange"] = display["PctChange"].round(1).astype(str) + "%"
            display["AnomalyScore"] = display["AnomalyScore"].round(3)
            display["Slope"] = display["Slope"].round(2)
            st.dataframe(display[["Description", "Slope", "PctChange", "RecentAvg", "Total", "AnomalyScore"]],
                         width='stretch', hide_index=True, height=360)
            st.download_button("⬇️ Download declining products (CSV)", data=declining.to_csv(index=False).encode("utf-8"),
                                file_name="declining_products.csv", mime="text/csv")

# ============================================================================
# PAGE: DATA EXPLORER
# ============================================================================
elif page == "🗂️ Data Explorer":
    section_title("Data Explorer", "Search, filter, and export the cleaned transaction-level data.")

    c1, c2, c3 = st.columns(3)
    search = c1.text_input("Search product description")
    cust_filter = c2.text_input("Filter by Customer ID")
    sort_by = c3.selectbox("Sort by", ["InvoiceDate", "Revenue", "Quantity", "Price"], index=0)

    view = dff.copy()
    if search:
        view = view[view["Description"].str.contains(search, case=False, na=False)]
    if cust_filter:
        try:
            view = view[view["Customer ID"] == int(cust_filter)]
        except ValueError:
            st.caption("⚠️ Customer ID must be numeric.")
    view = view.sort_values(sort_by, ascending=False)

    st.caption(f"Showing {len(view):,} of {len(dff):,} rows")
    st.dataframe(
        view[["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Revenue", "Customer ID", "Country"]].head(2000),
        width='stretch', hide_index=True, height=500,
    )
    st.download_button("⬇️ Download filtered data (CSV)", data=view.to_csv(index=False).encode("utf-8"),
                        file_name="filtered_transactions.csv", mime="text/csv")

# ============================================================================
# PAGE: MODEL CENTER
# ============================================================================
elif page == "🧠 Model Center":
    section_title("Model Center", "Every prediction in RetailIQ ML comes from one of the six models below, retrained live on your current filter selection.")

    rfm, silhouette = compute_rfm_kmeans(dff)
    models = train_customer_models(dff)
    fc = compute_forecast(dff)
    ar = compute_association_rules(dff)
    trend_feats = compute_product_trend_features(dff)

    model_card(
        "Customer Segmentation", "Unsupervised · Clustering",
        "K-Means (k=4) groups customers into Champions / Loyal / At Risk / Lapsed from log-transformed, standardized RFM features.",
        {"silhouette": f"{silhouette:.2f}" if silhouette is not None else "n/a", "customers": fmt_num(len(rfm))},
    )
    if models["ok"]:
        model_card(
            "Churn Prediction", "Supervised · Classification",
            f"Random Forest (300 trees) predicts whether a customer will make no purchase in the next {models['holdout_days']} days, trained with a strict time-based split to avoid leakage.",
            {"ROC-AUC": f"{models['churn_metrics']['roc_auc']:.3f}", "accuracy": f"{models['churn_metrics']['accuracy']*100:.1f}%",
             "train size": fmt_num(models["churn_metrics"]["n_train"])},
        )
        model_card(
            "Customer Lifetime Value", "Supervised · Regression",
            f"Gradient Boosting regressor predicts each customer's spend over the next {models['holdout_days']} days from current RFM behavior (trained on log1p-transformed spend).",
            {"MAE": fmt_gbp(models["clv_metrics"]["mae"]), "R² (log)": f"{models['clv_metrics']['r2_log']:.2f}"},
        )
    else:
        st.info(f"Churn / CLV models unavailable: {models['reason']}")

    if fc["ok"]:
        model_card(
            "Demand Forecasting", "Supervised · Time-Series Regression",
            "Linear regression on a trend index plus sine/cosine month-of-year features, backtested on the last 3 known months and projected 3 months forward with a 95% residual-based band.",
            {"backtest MAPE": f"{fc['mape']:.1f}%", "residual σ": fmt_gbp(fc["resid_std"])},
        )
    else:
        st.info(f"Forecasting model unavailable: {fc['reason']}")

    if ar["ok"]:
        model_card(
            "Product Recommendations", "Unsupervised · Association Rule Mining",
            "Apriori-style frequent-itemset mining: infrequent items are pruned before candidate pairs are generated, then each surviving pair is scored with support, confidence, and lift.",
            {"rules": fmt_num(len(ar["rules_df"])), "frequent items": fmt_num(ar["n_frequent_items"])},
        )
    else:
        st.info(f"Association rule model unavailable: {ar['reason']}")

    if not trend_feats.empty:
        model_card(
            "Decline / Anomaly Detection", "Unsupervised · Isolation Forest",
            "Isolation Forest flags products whose trend fingerprint (slope, volatility, % change, recent volume) is a statistical outlier; outliers with a negative slope are surfaced as declining.",
            {"products scored": fmt_num(len(trend_feats))},
        )

    st.write("")
    st.caption("Use **🔄 Retrain all models** in the sidebar to clear the cache and refit every model above on the current date range / country filters.")

# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
st.write("")
st.markdown(
    "<div style='text-align:center;color:#9ca3af;font-size:0.8rem;padding-top:10px;'>"
    "RetailIQ ML · Built with Streamlit &amp; scikit-learn · Dataset: Online Retail II (UCI ML Repository)"
    "</div>",
    unsafe_allow_html=True,
)
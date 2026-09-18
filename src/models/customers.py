import time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import FEATURE_COLS


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


@st.cache_data(show_spinner="Training churn (Random Forest) & CLV (Gradient Boosting) models…", max_entries=8, ttl=3600)
def train_customer_models(_df: pd.DataFrame, data_version: tuple = (), holdout_days: int = 90):
    _t0 = time.time()
    max_date = _df["InvoiceDate"].max()
    cutoff = max_date - pd.Timedelta(days=holdout_days)

    train_feats = compute_customer_features(_df, cutoff)
    if len(train_feats) < 40:
        return {"ok": False, "reason": "Not enough customer history in the current filter to train a reliable model (need a wider date range)."}

    holdout = _df[(_df["InvoiceDate"] > cutoff) & (_df["InvoiceDate"] <= max_date)]
    holdout_customers = set(holdout["Customer ID"].unique())
    holdout_spend = holdout.groupby("Customer ID")["Revenue"].sum()

    train_feats["Churned"] = (~train_feats.index.isin(holdout_customers)).astype(int)
    train_feats["FutureSpend"] = train_feats.index.map(holdout_spend).fillna(0.0)

    X = train_feats[FEATURE_COLS].fillna(0)
    y_churn = train_feats["Churned"]
    y_clv_log = np.log1p(train_feats["FutureSpend"])

    if y_churn.nunique() < 2:
        return {"ok": False, "reason": "Every customer fell into a single churn class in this filter selection — need a more varied sample."}

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

    churn_final = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, class_weight="balanced").fit(X, y_churn)
    clv_final = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42).fit(X, y_clv_log)

    now_feats = compute_customer_features(_df, max_date)
    Xnow = now_feats[FEATURE_COLS].fillna(0)
    now_feats["ChurnProb"] = churn_final.predict_proba(Xnow)[:, 1]
    now_feats["PredictedCLV"] = np.clip(np.expm1(clv_final.predict(Xnow)), 0, None)
    now_feats["RiskTier"] = pd.cut(now_feats["ChurnProb"], bins=[-0.01, 0.33, 0.66, 1.01], labels=["Low", "Medium", "High"])
    now_feats["PriorityScore"] = now_feats["ChurnProb"] * now_feats["PredictedCLV"]

    st.sidebar.caption(f"⏱️ train_customer_models compute: {time.time()-_t0:.2f}s")
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

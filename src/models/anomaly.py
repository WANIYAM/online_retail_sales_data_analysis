import time

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@st.cache_data(show_spinner="Engineering product trend features…", max_entries=8, ttl=3600)
def compute_product_trend_features(_df: pd.DataFrame, data_version: tuple = (), min_invoices: int = 15) -> pd.DataFrame:
    _t0 = time.time()
    txn_counts = _df.groupby("Description", observed=True)["Invoice"].nunique()
    eligible = txn_counts[txn_counts >= min_invoices].index
    sub = _df[_df["Description"].isin(eligible)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["Description", "Slope", "PctChange", "CV", "RecentAvg", "Total", "FirstHalfAvg"])

    medians = sub.groupby("Description", observed=True)["Quantity"].transform("median")
    sub["CappedQuantity"] = sub["Quantity"].clip(upper=medians * 5)
    sub["MonthP"] = sub["InvoiceDate"].dt.to_period("M")
    monthly_qty = sub.groupby(["Description", "MonthP"], observed=True)["CappedQuantity"].sum().reset_index()

    rows = []
    for desc, grp in monthly_qty.groupby("Description", observed=True):
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

    st.sidebar.caption(f"⏱️ compute_product_trend_features compute: {time.time()-_t0:.2f}s")
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

import time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.models.customers import compute_customer_features


@st.cache_data(show_spinner="Segmenting customers with RFM + K-Means…", max_entries=8, ttl=3600)
def compute_rfm_kmeans(_df: pd.DataFrame, data_version: tuple = ()):
    _t0 = time.time()
    max_date = _df["InvoiceDate"].max()
    rfm = compute_customer_features(_df, max_date)[["Recency", "Frequency", "Monetary"]]

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
    st.sidebar.caption(f"⏱️ compute_rfm_kmeans compute: {time.time()-_t0:.2f}s")
    return rfm.reset_index().rename(columns={"index": "Customer ID"}), silhouette

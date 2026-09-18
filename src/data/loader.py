import time
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner="Loading and cleaning transaction data…", max_entries=8, ttl=3600)
def load_data(source) -> pd.DataFrame:
    _t0 = time.time()
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
    for col in ["Description", "Country", "StockCode", "DayOfWeek"]:
        df_clean[col] = df_clean[col].astype("category")
    st.sidebar.caption(f"⏱️ load_data compute: {time.time()-_t0:.2f}s")
    return df_clean


@st.cache_data(show_spinner="Filtering transaction data…", max_entries=8, ttl=3600)
def filter_data(_df: pd.DataFrame, start_d, end_d, selected_countries: tuple) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_d)
    end_ts = pd.Timestamp(end_d) + pd.Timedelta(days=1)
    mask = (
        (_df["InvoiceDate"] >= start_ts)
        & (_df["InvoiceDate"] < end_ts)
        & (_df["Country"].isin(selected_countries))
    )
    return _df.loc[mask]

import time
import pandas as pd
import streamlit as st
from src.config import DATA_PATH
from src.data.loader import filter_data, load_data
from src.ui.styles import inject_styles

pg_overview = st.Page("pages/1_Overview.py", title="Overview", icon="📊")
pg_customer = st.Page("pages/2_Customer_Intelligence.py", title="Customer Intelligence", icon="🧭")
pg_forecast = st.Page("pages/3_Demand_Forecasting.py", title="Demand Forecasting", icon="📈")
pg_recs = st.Page("pages/4_Product_Recommendations.py", title="Product Recommendations", icon="🔗")
pg_anomaly = st.Page("pages/5_Anomaly_Detection.py", title="Anomaly Detection", icon="📉")
pg_explorer = st.Page("pages/6_Data_Explorer.py", title="Data Explorer", icon="🗂️")
pg_models = st.Page("pages/7_Model_Center.py", title="Model Center", icon="🧠")

DEFAULT_PAGES = [pg_overview, pg_customer, pg_forecast, pg_recs, pg_anomaly, pg_explorer, pg_models]


def render_sidebar_and_header(pages_list=None):
    if pages_list is None and "_run_cache" in st.session_state:
        cache = st.session_state["_run_cache"]
        return cache[1], cache[2]

    if pages_list is None:
        pages_list = DEFAULT_PAGES

    st.set_page_config(
        page_title="RetailIQ ML | Online Retail Intelligence",
        page_icon="🛍️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    app_t0 = time.time()

    with st.sidebar:
        st.markdown(
            "<div style='display:flex;align-items:center;gap:12px;margin-bottom:8px;"
            "padding:8px 0 4px 0;'>"
            "<span style='font-size:1.8rem;filter:drop-shadow(0 2px 10px rgba(253,121,168,0.7));'>🛍️</span>"
            "<span style='font-size:1.35rem;font-weight:900;font-family:Sora,sans-serif;"
            "letter-spacing:-0.03em;background:linear-gradient(120deg, #C4B5FD, #FD79A8, #00CEC9);"
            "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
            "background-clip:text;'>RetailIQ ML</span></div>",
            unsafe_allow_html=True,
        )
        st.caption("Online Retail Intelligence — Machine Learning Edition")
        st.divider()

        uploaded = st.file_uploader("Upload transactions CSV", type=["csv"], help="Defaults to the bundled Online Retail II dataset if left empty.")
        source = uploaded if uploaded is not None else (DATA_PATH if DATA_PATH.exists() else None)

        if source is None:
            st.error("No dataset found. Please upload a CSV with columns: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country.")
            st.stop()

        required_cols = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
        if uploaded is not None:
            try:
                header_cols = list(pd.read_csv(uploaded, nrows=0, encoding="ISO-8859-1").columns)
                if hasattr(uploaded, "seek"):
                    uploaded.seek(0)
                missing_cols = [col for col in required_cols if col not in header_cols]
                if missing_cols:
                    st.error(
                        f"Uploaded CSV is missing required column(s): **{', '.join(missing_cols)}**.\n\n"
                        f"Expected columns: `{', '.join(required_cols)}`"
                    )
                    st.stop()
            except Exception as e:
                st.error(f"Could not parse uploaded CSV header: {e}")
                st.stop()

        df_clean = load_data(source)

        st.divider()

        nav = st.navigation(pages_list, position="hidden")
        for p in pages_list:
            st.page_link(p, label=p.title, icon=p.icon)

        st.divider()

        st.markdown("**Filters**")
        min_d, max_d = df_clean["InvoiceDate"].min().date(), df_clean["InvoiceDate"].max().date()
        countries = sorted(df_clean["Country"].unique().tolist())

        if "filter_date_range" not in st.session_state:
            st.session_state["filter_date_range"] = (min_d, max_d)
        if "filter_countries" not in st.session_state:
            st.session_state["filter_countries"] = countries

        st.date_input("Date range", min_value=min_d, max_value=max_d, key="filter_date_range")
        st.multiselect("Countries", options=countries, key="filter_countries")

        st.divider()
        if st.button("🔄 Retrain all models", width='stretch', help="Clears cached models and retrains everything on the current filter selection."):
            st.cache_data.clear()
            st.rerun()

        st.caption(f"Loaded {len(df_clean):,} clean transaction rows")
        st.caption("Data: Online Retail II (UCI ML Repository)")

    date_range = st.session_state.get("filter_date_range", (min_d, max_d))
    selected_countries = st.session_state.get("filter_countries", countries)

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
    else:
        start_d, end_d = min_d, max_d

    selected_tuple = tuple(sorted(selected_countries if selected_countries else countries))
    dff = filter_data(df_clean, start_d, end_d, selected_tuple)

    data_version = (len(dff), str(start_d), str(end_d), selected_tuple)
    st.sidebar.caption(f"⏱️ App start to filter mask: {time.time()-app_t0:.3f}s")

    if dff.empty:
        st.warning("No data matches the current filters. Try widening the date range or country selection.")
        st.stop()

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

    result = (df_clean, dff, data_version, start_d, end_d, selected_countries, countries, nav)
    st.session_state["_run_cache"] = result
    return result

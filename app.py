"""
RetailIQ ML — Online Retail Intelligence Platform (Machine-Learning Edition)
=============================================================================
Main entrypoint for Streamlit application. Defines explicit st.Page list,
invokes custom sidebar layout, and executes selected page navigation.

Run with:  streamlit run app.py
"""

import streamlit as st
from src.ui.sidebar import render_sidebar_and_header

pg_overview = st.Page("pages/1_Overview.py", title="Overview", icon="📊")
pg_customer = st.Page("pages/2_Customer_Intelligence.py", title="Customer Intelligence", icon="🧭")
pg_forecast = st.Page("pages/3_Demand_Forecasting.py", title="Demand Forecasting", icon="📈")
pg_recs = st.Page("pages/4_Product_Recommendations.py", title="Product Recommendations", icon="🔗")
pg_anomaly = st.Page("pages/5_Anomaly_Detection.py", title="Anomaly Detection", icon="📉")
pg_explorer = st.Page("pages/6_Data_Explorer.py", title="Data Explorer", icon="🗂️")
pg_models = st.Page("pages/7_Model_Center.py", title="Model Center", icon="🧠")

pages_list = [pg_overview, pg_customer, pg_forecast, pg_recs, pg_anomaly, pg_explorer, pg_models]

res = render_sidebar_and_header(pages_list)
nav = res[-1]
nav.run()
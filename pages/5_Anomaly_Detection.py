import plotly.express as px
import streamlit as st

from src.config import PALETTE, PLOTLY_LAYOUT
from src.models.anomaly import compute_product_trend_features, run_isolation_forest
from src.ui.components import fmt_num, kpi_card, section_title
from src.ui.sidebar import render_sidebar_and_header

dff, data_version = render_sidebar_and_header()

section_title("Decline Detector — Isolation Forest", "Unsupervised anomaly detection over each product's sales-trend fingerprint (slope, volatility, % change, recent volume). Products flagged as statistical outliers with a negative trend are surfaced as at-risk.")

with st.expander("Detection settings"):
    c1, c2 = st.columns(2)
    min_invoices = c1.slider("Min. distinct orders (history requirement)", 5, 50, 15)
    contamination = c2.slider("Expected anomaly rate (contamination)", 0.02, 0.30, 0.10, 0.01)

trend_feats = compute_product_trend_features(dff, data_version, min_invoices=min_invoices)
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
        if st.button("Prepare download", key="btn_prep_declining"):
            st.session_state["prep_declining"] = True
        if st.session_state.get("prep_declining"):
            st.download_button("⬇️ Download declining products (CSV)", data=declining.to_csv(index=False).encode("utf-8"),
                                file_name="declining_products.csv", mime="text/csv")

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import PALETTE, PLOTLY_LAYOUT
from src.models.customers import train_customer_models
from src.models.segmentation import compute_rfm_kmeans
from src.ui.components import fmt_gbp, fmt_num, kpi_card, section_title
from src.ui.sidebar import render_sidebar_and_header

dff, data_version = render_sidebar_and_header()

rfm, silhouette = compute_rfm_kmeans(dff, data_version)
models = train_customer_models(dff, data_version)

tab_seg, tab_churn, tab_clv, tab_action = st.tabs(
    ["🧭 Segments (K-Means)", "⚠️ Churn Risk (Random Forest)", "💎 CLV Forecast (Gradient Boosting)", "🎯 Action Engine"]
)

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
        if st.button("Prepare download", key="btn_prep_action"):
            st.session_state["prep_action"] = True
        if st.session_state.get("prep_action"):
            st.download_button("⬇️ Download full customer action list (CSV)", data=merged.to_csv(index=False).encode("utf-8"),
                                file_name="customer_action_list.csv", mime="text/csv")

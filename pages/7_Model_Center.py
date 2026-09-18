import streamlit as st

from src.models.anomaly import compute_product_trend_features
from src.models.customers import train_customer_models
from src.models.forecasting import compute_forecast
from src.models.recommendations import compute_association_rules
from src.models.segmentation import compute_rfm_kmeans
from src.ui.components import fmt_gbp, fmt_num, model_card, section_title
from src.ui.sidebar import render_sidebar_and_header

dff, data_version = render_sidebar_and_header()

section_title("Model Center", "Every prediction in RetailIQ ML comes from one of the six models below, retrained live on your current filter selection.")

rfm, silhouette = compute_rfm_kmeans(dff, data_version)
models = train_customer_models(dff, data_version)
fc = compute_forecast(dff, data_version)
ar = compute_association_rules(dff, data_version)
trend_feats = compute_product_trend_features(dff, data_version)

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

st.write("")
st.markdown(
    "<div style='text-align:center;color:#9ca3af;font-size:0.8rem;padding-top:10px;'>"
    "RetailIQ ML · Built with Streamlit &amp; scikit-learn · Dataset: Online Retail II (UCI ML Repository)"
    "</div>",
    unsafe_allow_html=True,
)

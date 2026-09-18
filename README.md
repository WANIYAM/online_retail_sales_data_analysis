# RetailIQ ML — Online Retail Intelligence Platform (Machine Learning Edition)

A complete, polished Streamlit product built on the *Online Retail II* dataset. Every
analytical page is backed by a trained or fitted machine learning model — nothing is a
fixed business rule anymore.

## Models

| Page | Model | Type |
|---|---|---|
| Customer Intelligence — Segments | K-Means (k=4) on log/standardized RFM | Unsupervised clustering |
| Customer Intelligence — Churn Risk | Random Forest classifier | Supervised classification |
| Customer Intelligence — CLV Forecast | Gradient Boosting regressor (log1p target) | Supervised regression |
| Demand Forecasting | Linear regression on trend + sine/cosine seasonality | Supervised time-series regression |
| Product Recommendations | Apriori-style frequent-itemset mining (support/confidence/lift) | Unsupervised association-rule mining |
| Anomaly & Decline Detection | Isolation Forest over trend-fingerprint features | Unsupervised anomaly detection |

Churn and CLV are trained with a strict **time-based split** (features computed as of a
cutoff date, labels from the 90 days after it) so the model never sees the future during
training — then both models are refit on all available history and scored against every
currently active customer to drive the **Action Engine**, which ranks outreach by
`Churn Probability × Predicted Future Value`.

## Features

- **Overview** — KPI cards, monthly revenue with an ML forecast overlay, top products/countries, order patterns by weekday
- **Customer Intelligence** — segmentation, churn risk, CLV forecast, and a unified prioritized Action Engine with CSV export
- **Demand Forecasting** — trend + seasonality regression, backtested MAPE, 95% forecast band, model-learned seasonal index, forecast-based alerts
- **Product Recommendations** — Apriori-style "customers also bought" engine with support/confidence/lift and a searchable product picker
- **Anomaly & Decline Detection** — Isolation Forest flags products with an anomalous, declining sales trend, with adjustable sensitivity
- **Data Explorer** — searchable, filterable, downloadable view of the cleaned transaction data
- **Model Center** — a live registry of all six models with their current metrics, plus a one-click "Retrain all models" button
- Global sidebar filters (date range + country) apply across every page and every model is retrained against the filtered slice
- Clean custom-styled UI (KPI cards, gradient header, Plotly charts) — no extra UI dependencies required

## Setup

1. **Clone the repository and install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Download Dataset**:
   Download the full *Online Retail II* dataset (`online_retail_II.csv`) from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and place the `online_retail_II.csv` file directly in the project root directory.
   *(Note: A 50,000-row sample dataset is included at `data/sample_online_retail_II.csv` for lightweight testing, or you can upload custom transaction CSVs directly via the sidebar).*

## Run

```bash
streamlit run app.py
```

The app looks for `online_retail_II.csv` in the project root folder by default.
You can also upload a different CSV from the sidebar at any time — it must contain these
columns: `Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country`.

## Notes

- All expensive computations (model training, clustering, rule mining, anomaly detection)
  are cached with `st.cache_data`, so the app stays fast after the first load. Use the
  sidebar's **Retrain all models** button to clear the cache and refit everything on a new
  filter selection.
- If a filter selection leaves too little history to train a model reliably (e.g. one
  country, one month), the affected page shows a plain-language notice instead of a
  misleading result.
- The final month of data is partial (the dataset ends mid-month); the forecasting model
  automatically excludes it from fitting so the trend isn't skewed by an incomplete month.
- Currency is displayed in GBP (£), matching the dataset's UK-based retailer.
- Data source: [Online Retail II, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii).

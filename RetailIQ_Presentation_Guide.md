# RetailIQ ML — Presentation Guide
### A plain-English walkthrough of every screen, model, and calculation

Use this as your speaker notes. Each section explains **what the audience sees**, **why it
matters to the business**, and **which machine learning model produced it** — with any
technical term defined the first time it shows up.

---

## 1. The Big Picture (30-second pitch)

> "We took over a million raw sales transactions from an online retailer, cleaned them up,
> and trained six machine learning models on them — a clustering model, two prediction
> models, a forecasting model, a pattern-mining model, and an anomaly detector. Together
> they answer five business questions automatically: *How are we doing? Who's about to
> churn, and what are they worth? When will demand spike? What do people buy together?
> Which products are anomalously declining?*"

The app is built with **Streamlit** — a Python tool that turns a data-analysis script into
a clickable web app, without needing separate web-development work. Every chart is
interactive (hover, zoom) and built with **Plotly**, a charting library. Every prediction
comes from a model trained with **scikit-learn**, the standard Python machine learning
library.

---

## 2. How Data Flows Through the App (architecture, in plain terms)

Think of it like a kitchen, now with a machine-learning line cook added to the prep station:

1. **Raw ingredients (`online_retail_II.csv`)** — the original spreadsheet of every single item sold: invoice number, product, quantity, price, date, customer, country.
2. **Prep station (data cleaning)** — before any analysis happens, the app throws out unusable rows:
   - Rows with no **Customer ID** or no product description.
   - **Cancellations** — orders where the invoice number starts with "C" (a returned/cancelled order).
   - Rows with zero or negative quantity/price (data errors or refund adjustments).
3. **The result is `df_clean`** — a trustworthy table of only real, completed, positive sales, with a **Revenue** column (`Quantity × Price`).
4. **Feature engineering** — for every customer and every product, the app computes summary numbers (how recently they bought, how often, how much, how their sales are trending) that the models actually learn from. This is the raw material every model below is built on.
5. **Model training** — six models are fit on those features (details in Section 3). Two of them (Churn and CLV) are trained with a **time-based split**: the model only ever sees data up to a cutoff date while training, and is graded on what actually happened afterward — the same discipline a real forecasting team would use, so the reported accuracy isn't inflated by hindsight.
6. **Caching** — the app remembers the result of a slow calculation (including model training) so it doesn't have to redo it every time you click something. (Technical: `st.cache_data`, Streamlit's memoization feature.) The sidebar's **Retrain all models** button clears this cache on demand.
7. **Filters (sidebar)** — Date range and Country selectors narrow the data for *every single page*, and every model is refit against just that slice.

---

## 3. Page-by-Page Walkthrough

### 📊 Overview — "How is the business doing right now?"

**What's on screen:** Five KPI cards, a monthly revenue chart with the forecasting model's
projection overlaid as a dashed line and shaded confidence band, top 10 products/countries,
and order volume by weekday.

**Why it matters:** The "dashboard cockpit" — a 5-second gut-check on business health,
now with a forward-looking view baked in rather than just a look backward.

---

### 🧭 Customer Intelligence — "Who are our customers, who's leaving, and what are they worth?"

This page has four tabs, each backed by its own model.

**Tab 1 — Segments (K-Means).** Three numbers per customer — **Recency** (days since last
order), **Frequency** (number of orders), **Monetary** (total spend) — are log-transformed
and standardized, then fed into **K-Means**, an unsupervised clustering algorithm ("unsupervised"
means it isn't told the right answer in advance; it finds natural groupings on its own). It
sorts customers into 4 clusters, which the app labels Champions / Loyal / At Risk / Lapsed
based on their average behavior. A **silhouette score** (−1 to 1) is shown as a report card
on how cleanly separated those clusters actually are.

**Tab 2 — Churn Risk (Random Forest).** A **Random Forest** — an ensemble of many decision
trees that vote together — is trained to predict whether a customer will make *no* purchase
in the next 90 days, using their Recency/Frequency/Monetary and a few other features as of
90 days before the end of the data. It's graded on a **held-out test set** of customers it
never saw during training. The headline number is **ROC-AUC** (a 0.5–1.0 score measuring how
well the model ranks "will churn" customers above "won't churn" customers; 1.0 is perfect,
0.5 is a coin flip). A feature-importance chart shows which inputs the model leaned on most.

**Tab 3 — CLV Forecast (Gradient Boosting).** A **Gradient Boosting** regressor (another
tree-ensemble method, built to correct its own mistakes round by round) predicts how much
each customer will actually spend in the next 90 days. Accuracy is reported as **MAE**
(Mean Absolute Error — the average £ the model is off by) and **R²** (how much of the
variation in spending the model explains; closer to 1 is better).

**Tab 4 — Action Engine.** Combines all three: each customer's segment, churn probability,
and predicted future value are merged into one table, ranked by
**Priority Score = Churn Probability × Predicted Future Value** — in plain terms, "how much
revenue is genuinely at risk of walking out the door." This is the to-do list handed to
marketing, downloadable as a CSV.

**One-liner for the audience:** *"Instead of guessing who's about to leave, we trained a
model to score every customer's churn risk and future value, then multiplied the two to
find where the real money is at stake."*

---

### 📈 Demand Forecasting — "When should we prepare for a rush, and how confident are we?"

**How it's calculated:** A **linear regression** model is trained on monthly revenue,
using a trend number (month 1, 2, 3…) plus a sine/cosine encoding of the calendar month
(a standard way to teach a model "this repeats every 12 months" without hard-coding each
month as a separate category). The model is **backtested** — trained on all but the last
three known months, then asked to predict those three — and graded with **MAPE** (Mean
Absolute Percentage Error), so the audience sees exactly how far off the model's own recent
predictions were before trusting its forecast of the future. The forecast comes with a 95%
confidence band built from the model's residual error.

**What's on screen:** Actual vs. fitted vs. forecast chart, a "model-learned seasonality"
bar chart (the regression's own seasonal coefficients, re-expressed as an index around 1.0
so it reads like the familiar "December is busy" chart — except now it comes straight out
of a trained model instead of a simple average), and forecast-based alerts for the next
three months.

**Data quality note worth mentioning:** the dataset's last month is a partial month (it
cuts off mid-December), so the model automatically excludes it from fitting — otherwise
an artificially low partial month would drag the whole trend down.

---

### 🔗 Product Recommendations — "What do people buy together, and how confident should we be?"

This is **Apriori-style association rule mining** — the same family of algorithm behind
Amazon's "customers who bought this also bought…" The pruning idea (Apriori's core trick)
is: **first drop products that don't appear often enough on their own**, so the number of
candidate pairs to check stays manageable, then count how often each surviving pair
appears together.

Each pair is scored three ways:
- **Support** — how often the pair appears, as a share of all baskets.
- **Confidence** — of the baskets with product A, what share also had product B.
- **Lift** — how much more often the pair co-occurs than pure chance would predict. Lift
  above 1 means a real association, not just two popular items that happen to show up a lot.

**Why it matters:** the same cross-sell logic as before, but now backed by textbook
data-mining metrics instead of a raw co-occurrence count — so "these two items are linked"
comes with a confidence number attached.

---

### 📉 Anomaly & Decline Detection — "What's dying out, statistically speaking?"

**How it's calculated:** For every product with enough sales history, the app builds a
"trend fingerprint" — its month-over-month sales slope, volatility, % change from the first
half of its history to the second, and recent volume. An **Isolation Forest** — an
unsupervised algorithm that isolates unusual data points faster than typical ones, the way
an oddly-shaped puzzle piece gets separated out sooner — scores every product for how
statistically unusual its fingerprint is. Products that are *both* unusual *and* trending
downward are flagged as declining; a slider controls how strict ("contamination rate") the
detector should be.

**Why it matters:** this replaces a single fixed "40% drop" rule with a model that adapts
to what's actually unusual in the current data, and separates genuine outliers from normal
month-to-month noise.

---

### 🗂️ Data Explorer — "Let me look at the raw receipts myself"

Unchanged — a searchable, filterable, exportable view of the individual cleaned
transactions. This is the "trust but verify" page.

---

### 🧠 Model Center — "Show me the models, not just the answers"

A registry page listing all six models, their type (supervised/unsupervised,
classification/regression/clustering/mining/anomaly detection), a one-line description of
what they do, and their live, current-filter performance metrics. A **Retrain all models**
button in the sidebar clears the cache and refits everything from scratch — useful when you
change the date range or country filter and want fresh numbers, not stale cached ones.

---

## 4. Glossary — Technical Terms in Plain English

| Term | Plain-English Definition |
|---|---|
| **Streamlit** | A Python tool that turns a data script into an interactive web app without writing HTML/JavaScript. |
| **Plotly** | The charting library used for all the interactive graphs. |
| **scikit-learn** | The Python machine learning library that trains every model in this app. |
| **Caching** | Saving the result of a slow calculation (including model training) so it doesn't need to be redone every time — makes the app feel fast. |
| **K-Means Clustering** | An unsupervised method that automatically groups similar data points together. |
| **RFM** | Recency, Frequency, Monetary — three numbers summarizing a customer's buying behavior. |
| **Silhouette Score** | A −1 to 1 score for how cleanly separated a clustering's groups are. |
| **Random Forest** | A supervised classification method that combines many decision trees' votes into one prediction. |
| **Gradient Boosting** | A supervised method that builds trees one at a time, each correcting the last one's errors. |
| **Time-Based Split** | Training a model only on data up to a cutoff date, then testing it against what actually happened after — avoids "cheating" with future information. |
| **ROC-AUC** | A 0.5–1.0 score for how well a classifier ranks positive cases above negative ones; 1.0 is perfect, 0.5 is a coin flip. |
| **MAE (Mean Absolute Error)** | The average size of a regression model's prediction error, in the original units (e.g., £). |
| **R² (R-squared)** | The share of variation in the outcome that a regression model explains; closer to 1 is better. |
| **MAPE (Mean Absolute Percentage Error)** | Forecast error expressed as a percentage, from a backtest against known months. |
| **Apriori / Association Rule Mining** | A method for finding which items tend to appear together, using support, confidence, and lift. |
| **Support, Confidence, Lift** | Support = how often a pair occurs; confidence = how often B follows A; lift = how much more likely than chance. |
| **Isolation Forest** | An unsupervised method that flags data points as anomalies by how quickly they can be isolated from the rest. |
| **Feature Importance** | A ranking of which inputs a trained model relied on most for its predictions. |
| **Churn** | A customer going inactive / not returning to buy again. |
| **CLV (Customer Lifetime Value)** | The (here: near-term predicted) revenue a customer is expected to generate. |
| **Seasonal Index** | A number showing how a given month's sales compare to the yearly average (here, derived from the forecasting model's own seasonal coefficients). |
| **KPI (Key Performance Indicator)** | A single important number used to judge business health at a glance. |
| **AOV (Average Order Value)** | Total revenue divided by number of orders. |
| **CSV** | "Comma-Separated Values" — a plain text spreadsheet format. |

---

## 5. Suggested Talking Points / Analogies

- **Churn + CLV together:** *"A model that only flags who's leaving is only half the story — we multiply that risk by what they're actually worth, so the list surfaces where the real money is, not just the most customers."*
- **Time-based split:** *"We graded the model the honest way — it only ever saw the past during training, then we checked it against what customers actually did next, the same test a real analyst would face."*
- **Apriori pruning:** *"Before counting every possible pair of a few thousand products, we first throw out anything too rare to matter — that's the trick that makes this fast."*
- **Isolation Forest:** *"Instead of a fixed '40% drop counts as a problem' rule, we let the algorithm learn what 'normal' looks like in the current data and flag what stands out."*
- **Forecast confidence band:** *"We don't just give one number for next month — we show the range the model has actually been wrong by in the past, so you know how much to trust it."*

---

## 6. Anticipated Audience Questions

**Q: How do we know the churn/CLV models aren't just overfitting or cheating with hindsight?**
A: They're trained with a time-based split — the model only sees data up to a cutoff date during training, and is graded on a genuinely held-out set of customers and a genuinely later time window. The reported ROC-AUC, accuracy, MAE, and R² all come from that held-out evaluation, not from data the model trained on.

**Q: Why Random Forest for churn but Gradient Boosting for CLV?**
A: Random Forest is a strong, stable baseline for classification with modest tuning; Gradient Boosting tends to fit skewed numeric targets (like spend, which has a long tail of big spenders) a bit better, especially once we log-transform the target. Both are standard, well-understood scikit-learn models — this isn't a black box.

**Q: Does the association-rule engine need a separate library like mlxtend?**
A: No — it's implemented directly with the Apriori pruning idea (drop infrequent items before generating pairs) and the standard support/confidence/lift formulas, so it stays inside the existing dependency footprint.

**Q: Does this update in real time?**
A: It runs on an uploaded/bundled CSV snapshot and retrains on demand via the sidebar button. It could be connected to a live database or a scheduled retraining job as a next step.

**Q: What happens if I change the date range or country filter?**
A: Every model on every page is refit against just that filtered slice. If the filtered slice is too small to train a model reliably (e.g., one country, one month), the page shows a plain-language notice instead of an unreliable number.

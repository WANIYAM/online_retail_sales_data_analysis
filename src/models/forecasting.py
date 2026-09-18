import time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression


@st.cache_data(show_spinner="Fitting demand forecasting model…", max_entries=8, ttl=3600)
def compute_forecast(_df: pd.DataFrame, data_version: tuple = (), n_forecast: int = 3):
    _t0 = time.time()
    monthly_rev = _df.groupby("MonthYear")["Revenue"].sum().sort_index()
    if len(monthly_rev) < 8:
        return {"ok": False, "reason": "Need at least 8 months of history in the current filter to fit a seasonal model."}

    max_date = _df["InvoiceDate"].max()
    last_period = monthly_rev.index[-1]
    days_in_last = (last_period.end_time - last_period.start_time).days + 1
    days_elapsed = (max_date - last_period.start_time).days + 1
    is_partial = days_elapsed < days_in_last - 1

    fit_series = monthly_rev.iloc[:-1] if is_partial else monthly_rev
    if len(fit_series) < 7:
        return {"ok": False, "reason": "Not enough complete months of history to fit a seasonal model."}

    def build_X(series):
        idx = np.arange(len(series))
        months = series.index.month.values
        return np.column_stack([idx, np.sin(2 * np.pi * months / 12), np.cos(2 * np.pi * months / 12)])

    Xf = build_X(fit_series)
    yf = fit_series.values

    n_back = min(3, len(fit_series) - 4)
    lr_bt = LinearRegression().fit(Xf[:-n_back], yf[:-n_back])
    pred_bt = lr_bt.predict(Xf[-n_back:])
    actual_bt = yf[-n_back:]
    mape = float(np.mean(np.abs((actual_bt - pred_bt) / np.where(actual_bt == 0, 1, actual_bt))) * 100)

    lr_full = LinearRegression().fit(Xf, yf)
    fitted = lr_full.predict(Xf)
    resid_std = float(np.std(yf - fitted))

    future_periods = [fit_series.index[-1] + i for i in range(1, n_forecast + 1)]
    future_idx = np.arange(len(fit_series), len(fit_series) + n_forecast)
    fm = np.array([p.month for p in future_periods])
    Xfut = np.column_stack([future_idx, np.sin(2 * np.pi * fm / 12), np.cos(2 * np.pi * fm / 12)])
    forecast = lr_full.predict(Xfut)

    trend_mean = Xf[:, 0].mean()
    month_effects = {}
    for m in range(1, 13):
        xm = np.array([[trend_mean, np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)]])
        month_effects[m] = lr_full.predict(xm)[0]
    avg_effect = np.mean(list(month_effects.values()))
    seasonal_index = pd.Series({m: v / avg_effect for m, v in month_effects.items()}).reindex(range(1, 13))

    st.sidebar.caption(f"⏱️ compute_forecast compute: {time.time()-_t0:.2f}s")
    return {
        "ok": True,
        "monthly_rev": monthly_rev,
        "fit_series": fit_series,
        "is_partial_last_month": is_partial,
        "fitted": pd.Series(fitted, index=fit_series.index),
        "backtest_actual": pd.Series(actual_bt, index=fit_series.index[-n_back:]),
        "backtest_pred": pd.Series(pred_bt, index=fit_series.index[-n_back:]),
        "mape": mape,
        "future_periods": future_periods,
        "forecast": forecast,
        "resid_std": resid_std,
        "seasonal_index": seasonal_index,
    }

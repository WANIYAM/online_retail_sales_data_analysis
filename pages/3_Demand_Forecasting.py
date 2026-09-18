import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import ACCENT, ACCENT_2, PALETTE, PLOTLY_LAYOUT
from src.models.forecasting import compute_forecast
from src.ui.components import fmt_gbp, kpi_card, section_title
from src.ui.sidebar import render_sidebar_and_header

dff, data_version = render_sidebar_and_header()

fc = compute_forecast(dff, data_version)
section_title("Demand Forecasting — Trend + Seasonality Regression", "A linear regression model learns a revenue trend plus a seasonal (month-of-year) effect from history, then projects it forward.")

if not fc["ok"]:
    st.info(fc["reason"])
else:
    c1, c2, c3 = st.columns(3)
    kpi_card(c1, "Backtest MAPE", f"{fc['mape']:.1f}%", "🎯", color=PALETTE[0], sub="Error when the model predicts the last 3 known months")
    kpi_card(c2, f"Forecast — {fc['future_periods'][0]}", fmt_gbp(fc["forecast"][0]), "🔮", color=PALETTE[1])
    kpi_card(c3, "Residual Std. Dev.", fmt_gbp(fc["resid_std"]), "📏", color=PALETTE[4], sub="Used for the 95% forecast band")

    st.write("")
    section_title("Actual vs. Fitted vs. Forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc["fit_series"].index.astype(str), y=fc["fit_series"].values, mode="markers+lines",
                              name="Actual", line=dict(color=ACCENT, width=2)))
    fig.add_trace(go.Scatter(x=fc["fitted"].index.astype(str), y=fc["fitted"].values, mode="lines",
                              name="Model fit", line=dict(color="#9ca3af", width=2, dash="dot")))
    future_x = [str(p) for p in fc["future_periods"]]
    bridge_x = [str(fc["fit_series"].index[-1])] + future_x
    bridge_y = [fc["fit_series"].values[-1]] + list(fc["forecast"])
    upper = np.array(bridge_y) + 1.96 * fc["resid_std"]
    lower = np.array(bridge_y) - 1.96 * fc["resid_std"]
    fig.add_trace(go.Scatter(x=bridge_x + bridge_x[::-1], y=list(upper) + list(lower[::-1]),
                              fill="toself", fillcolor="rgba(0,206,201,0.15)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=bridge_x, y=bridge_y, mode="lines+markers", name="Forecast",
                              line=dict(color=ACCENT_2, width=3, dash="dash")))
    if fc["is_partial_last_month"]:
        partial = fc["monthly_rev"].iloc[-1:]
        fig.add_trace(go.Scatter(x=partial.index.astype(str), y=partial.values, mode="markers",
                                  name="Partial (in progress)", marker=dict(color="#E17055", size=10, symbol="x")))
    fig.update_layout(**PLOTLY_LAYOUT, height=380, yaxis_title="Revenue (£)")
    st.plotly_chart(fig, width='stretch')

    col1, col2 = st.columns([1, 2])
    with col1:
        section_title("Model-Learned Seasonality", "The regression's seasonal component per calendar month (1.0 = average).")
        month_names = [pd.to_datetime(f"2024-{m}-01").month_name()[:3] for m in range(1, 13)]
        si = fc["seasonal_index"]
        colors = ["#E17055" if v and v > 1.15 else ("#00B894" if v and v < 0.9 else ACCENT) for v in si.values]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=month_names, y=si.values, marker_color=colors))
        fig2.add_hline(y=1.0, line_dash="dash", line_color="#9ca3af")
        fig2.update_layout(**PLOTLY_LAYOUT, height=320, yaxis_title="Seasonal Index")
        st.plotly_chart(fig2, width='stretch')
    with col2:
        section_title("Forecast Alerts", "Flags months where the model's forecast is meaningfully above the trailing 12-month average.")
        threshold = st.slider("Alert threshold (× trailing 12-month avg revenue)", 1.0, 2.0, 1.2, 0.05)
        trailing_avg = fc["monthly_rev"].iloc[-12:].mean() if len(fc["monthly_rev"]) >= 3 else fc["monthly_rev"].mean()
        any_alert = False
        for p, v in zip(fc["future_periods"], fc["forecast"]):
            ratio = v / trailing_avg if trailing_avg else 0
            if ratio > threshold:
                any_alert = True
                st.warning(f"**ALERT:** the model forecasts **{fmt_gbp(v)}** for {p} — **{(ratio-1)*100:.0f}% above** the trailing 12-month average. Plan inventory and staffing ahead of time.")
        if not any_alert:
            st.success("No forecasted months exceed the alert threshold — demand looks steady.")

    with st.expander("View raw monthly revenue series"):
        st.dataframe(fc["monthly_rev"].rename("Revenue (£)").to_frame().reset_index().rename(columns={"MonthYear": "Month"}), width='stretch', hide_index=True)

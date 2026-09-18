import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import ACCENT, ACCENT_2, PALETTE, PLOTLY_LAYOUT
from src.models.forecasting import compute_forecast
from src.ui.components import fmt_gbp, fmt_num, kpi_card, section_title
from src.ui.sidebar import render_sidebar_and_header

dff, data_version = render_sidebar_and_header()

total_rev = dff["Revenue"].sum()
total_orders = dff["Invoice"].nunique()
total_customers = dff["Customer ID"].nunique()
aov = total_rev / total_orders if total_orders else 0
total_units = dff["Quantity"].sum()

c1, c2, c3, c4, c5 = st.columns(5)
kpi_card(c1, "Total Revenue", fmt_gbp(total_rev), "💰", color=PALETTE[0])
kpi_card(c2, "Orders", fmt_num(total_orders), "🧾", color=PALETTE[1])
kpi_card(c3, "Customers", fmt_num(total_customers), "👥", color=PALETTE[2])
kpi_card(c4, "Avg Order Value", fmt_gbp(aov), "🛒", color=PALETTE[3])
kpi_card(c5, "Units Sold", fmt_num(total_units), "📦", color=PALETTE[4])

st.write("")
section_title("Monthly Revenue — Actual vs. ML Forecast", "Solid line is actual revenue; dashed line and shaded band are the trend+seasonality regression model's forecast.")
fc = compute_forecast(dff, data_version)
if fc["ok"]:
    hist = fc["monthly_rev"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index.astype(str), y=hist.values, mode="lines+markers",
                              name="Actual", line=dict(color=ACCENT, width=3), fill="tozeroy", fillcolor="rgba(108,92,231,0.10)"))
    future_x = [str(p) for p in fc["future_periods"]]
    bridge_x = [str(fc["fit_series"].index[-1])] + future_x
    bridge_y = [fc["fit_series"].values[-1]] + list(fc["forecast"])
    upper = np.array(bridge_y) + 1.96 * fc["resid_std"]
    lower = np.array(bridge_y) - 1.96 * fc["resid_std"]
    fig.add_trace(go.Scatter(x=bridge_x + bridge_x[::-1], y=list(upper) + list(lower[::-1]),
                              fill="toself", fillcolor="rgba(0,206,201,0.15)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=bridge_x, y=bridge_y, mode="lines+markers", name="Forecast (95% band)",
                              line=dict(color=ACCENT_2, width=3, dash="dash")))
    fig.update_layout(**PLOTLY_LAYOUT, height=360, yaxis_title="Revenue (£)", xaxis_title=None)
    st.plotly_chart(fig, width='stretch')
    note = " (latest month is partial and excluded from model fitting)" if fc["is_partial_last_month"] else ""
    st.caption(f"Backtest MAPE: {fc['mape']:.1f}%{note} — see the Demand Forecasting page for details.")
else:
    st.info(fc["reason"])

col1, col2 = st.columns(2)
with col1:
    section_title("Top 10 Products by Revenue")
    top_products = dff.groupby("Description", observed=True)["Revenue"].sum().sort_values(ascending=False).head(10).sort_values()
    fig = px.bar(top_products, x=top_products.values, y=top_products.index, orientation="h",
                 labels={"x": "Revenue (£)", "y": ""}, color_discrete_sequence=[PALETTE[0]])
    fig.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False)
    st.plotly_chart(fig, width='stretch')
with col2:
    section_title("Top 10 Countries by Revenue")
    top_countries = dff.groupby("Country", observed=True)["Revenue"].sum().sort_values(ascending=False).head(10).sort_values()
    fig = px.bar(top_countries, x=top_countries.values, y=top_countries.index, orientation="h",
                 labels={"x": "Revenue (£)", "y": ""}, color_discrete_sequence=[PALETTE[1]])
    fig.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False)
    st.plotly_chart(fig, width='stretch')

section_title("Order Volume by Day of Week")
days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
orders_by_day = dff.groupby("DayOfWeek", observed=True)["Invoice"].nunique().reindex(days_order).fillna(0)
fig = px.bar(orders_by_day, x=orders_by_day.index, y=orders_by_day.values,
             labels={"x": "", "y": "Unique Orders"}, color_discrete_sequence=[PALETTE[3]])
fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
st.plotly_chart(fig, width='stretch')

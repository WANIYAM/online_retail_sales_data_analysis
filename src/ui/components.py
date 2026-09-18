import streamlit as st
from src.config import ACCENT


def fmt_gbp(x: float) -> str:
    if abs(x) >= 1_000_000:
        return f"£{x/1_000_000:,.2f}M"
    if abs(x) >= 1_000:
        return f"£{x/1_000:,.1f}K"
    return f"£{x:,.0f}"


def fmt_num(x: float) -> str:
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:,.2f}M"
    if abs(x) >= 1_000:
        return f"{x/1_000:,.1f}K"
    return f"{x:,.0f}"


def kpi_card(col, label, value, icon, sub=None, color=ACCENT):
    with col:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-left:4px solid {color}; --kpi-glow: {color}1a;">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                {f'<div class="kpi-sub">{sub}</div>' if sub else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )


def section_title(title, sub=None):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if sub:
        st.markdown(f"<div class='section-sub'>{sub}</div>", unsafe_allow_html=True)


def risk_badge(tier: str) -> str:
    cls = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(tier, "badge-low")
    return f"<span class='badge {cls}'>{tier}</span>"


def model_card(name, mtype, desc, metrics: dict):
    metric_html = "".join(f"<span class='model-metric'><b>{v}</b> {k}</span>" for k, v in metrics.items())
    st.markdown(
        f"""
        <div class="model-card">
            <span class="model-name">{name}</span><span class="model-type">{mtype}</span>
            <div class="model-desc">{desc}</div>
            <div>{metric_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

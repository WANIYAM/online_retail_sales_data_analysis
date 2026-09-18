from pathlib import Path
import streamlit as st

STYLES_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "styles.css"


def inject_styles():
    css = STYLES_PATH.read_text(encoding="utf-8")
    st.markdown(
        f"""
        <style>
        {css}
        </style>
        """,
        unsafe_allow_html=True,
    )

# app.py — Ultra Pro Max landing page (root)
import streamlit as st
from utils import load_sample_csv, download_link_html

st.set_page_config(page_title="Credit Card Fraud — Ultra Pro Max", layout="wide", page_icon="💳")

# top-level styling (Ultra Pro Max)
st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg,#07101a 0%, #04121a 100%); color: #e6eef6; }
      .page-title { font-size: 32px; font-weight:700; color:#cfe9ff; margin-bottom:6px; }
      .muted { color:#9aa6b2; margin-bottom:12px; }
      .card { background: rgba(255,255,255,0.02); padding:12px; border-radius:10px; border: 1px solid rgba(80,160,255,0.04); }
      a { color:#7fd1ff; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="page-title">💳 Credit Card Fraud Detection — Ultra Pro Max</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">A multi-page premium app. Use the left sidebar (Pages) to navigate.</div>', unsafe_allow_html=True)

st.markdown("---")

# Load sample for quick access
try:
    df = load_sample_csv()
    st.markdown("**Included sample dataset (1000 rows)** — quick demo")
    st.markdown(download_link_html(df), unsafe_allow_html=True)
    st.caption("Pages included: Dashboard • Model Performance • Class Distribution • Feature Importances • Manual Prediction • Resources")
except FileNotFoundError:
    st.error("sample_creditcard.csv not found in repo root. Add it and reboot, or upload on Resources page.")

st.markdown("---")
st.caption("Made with ❤️ — Ultra Pro Max B (futuristic theme)")

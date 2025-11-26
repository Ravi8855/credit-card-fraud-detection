import streamlit as st
from utils import load_sample_csv, download_link_html, train_model_pipeline

st.set_page_config(page_title="Credit Card Fraud — Ultra Pro", layout="wide", page_icon="💳")

# Top-level CSS to unify the neon-blue cyber theme
st.markdown(
    """
    <style>
    body { background: #070b10; color: #e6eef6; }
    .stApp { background: linear-gradient(180deg,#07101a 0%, #04121a 100%); }
    .page-title { font-size: 34px; font-weight:700; color:#cfe9ff; }
    .muted { color:#9aa6b2; }
    .card { background: rgba(255,255,255,0.02); padding:14px; border-radius:10px; border: 1px solid rgba(10,140,255,0.06);}
    a { color:#7fd1ff; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown('<div class="page-title">💳 Credit Card Fraud Detection — Ultra Pro Max</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Use the left sidebar (Streamlit `pages`) to navigate between pages.</div>', unsafe_allow_html=True)
st.markdown("---")

# Try to load sample CSV so that app boots fast
try:
    df = load_sample_csv()

    st.markdown("**Pages included:**")
    st.write("• Dashboard • Model Performance • Class Distribution • Feature Importances • Manual Prediction • Resources")

    # Download link
    st.markdown(download_link_html(df), unsafe_allow_html=True)

except Exception as e:
    st.error("sample_creditcard.csv missing. Add it to repo root or upload it in Resources page.")

st.caption("Made with ❤️ — Ultra Pro Max Theme A")

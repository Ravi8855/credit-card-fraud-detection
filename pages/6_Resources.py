# pages/6_Resources.py
import streamlit as st
from utils import load_sample_csv, download_link_html, save_trained_model, train_model_pipeline

st.set_page_config(page_title="Resources", layout="wide")
st.markdown("## 📦 Resources & Model Export")

st.markdown("### Upload dataset (optional)")
uploaded = st.file_uploader("Upload CSV (single row or full dataset)", type=["csv"])
if uploaded is not None:
    st.success("File uploaded — you can use it for training (not persisted on Streamlit Cloud long-term).")

# show sample dataset download and option to export model
try:
    df = load_sample_csv()
    st.markdown(download_link_html(df), unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("sample_creditcard.csv not in repo root. Add it or upload above.")

if st.button("Retrain model (force)"):
    try:
        df = load_sample_csv()
        metrics = train_model_pipeline(df)
        path = save_trained_model("trained_model_and_scaler.joblib", metrics["model"], metrics["scaler"])
        st.success(f"Saved {path}")
    except Exception as e:
        st.error(f"Retrain failed: {e}")

st.markdown("---")
st.write("Quick links:")
st.write("- GitHub repo (open in a new tab after deploy)")
st.write("- Original dataset: not included in repo (too large).")
st.caption("Made with ❤️ — Ultra Pro Max (B)")

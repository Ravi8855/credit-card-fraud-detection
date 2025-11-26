# pages/6_Resources.py
import streamlit as st
from utils import load_sample_csv, download_link_html

st.set_page_config(layout="wide")
st.title("📦 Resources")

try:
    df = load_sample_csv()
    st.markdown(download_link_html(df), unsafe_allow_html=True)
except Exception:
    st.info("No sample CSV available in repo root. Add 'sample_creditcard.csv' to enable download.")

st.markdown("""
### Helpful links
- Put your GitHub repository link here
- Put any documentation or paper links here
""")

# pages/1_Dashboard.py
import streamlit as st
import pandas as pd
from utils import load_sample_csv, download_link_html, ensure_feature_order

st.set_page_config(layout="wide")

st.title("📊 Dashboard")

try:
    df = load_sample_csv()
except FileNotFoundError:
    st.error("sample_creditcard.csv not found in repo root. Add a file named 'sample_creditcard.csv' and reboot the app.")
    st.stop()

df = ensure_feature_order(df)

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{df.shape[0]:,}")
c2.metric("Features", f"{df.shape[1]-1}")
c3.metric("Target column", "Class")

if st.checkbox("Show raw data (first 50 rows)"):
    st.dataframe(df.head(50), use_container_width=True)

st.markdown(download_link_html(df), unsafe_allow_html=True)

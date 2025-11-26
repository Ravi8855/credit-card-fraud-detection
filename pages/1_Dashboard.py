import streamlit as st
import pandas as pd
from utils import load_sample_csv, ensure_feature_order


st.set_page_config(page_title="Dashboard", layout="wide")


st.markdown("# 📊 Dashboard")


try:
df = load_sample_csv()
except FileNotFoundError:
st.error("sample_creditcard.csv not found. Add the file to repo root.")
st.stop()


cols = ensure_feature_order(df)


c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{df.shape[0]:,}")
c2.metric("Features", f"{len(cols) - (1 if 'Class' in cols else 0)}")
c3.metric("Target", "Class")


if st.checkbox("Show first 50 rows"):
st.dataframe(df.head(50))
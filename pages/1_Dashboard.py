import streamlit as st
import pandas as pd
from utils import load_sample_csv

st.title("🏠 Dashboard")

df = load_sample_csv()

col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{df.shape[0]:,}")
col2.metric("Features", df.shape[1] - 1)
col3.metric("Target Column", "Class")

if st.checkbox("Show sample data"):
    st.dataframe(df.head(50))

# pages/3_Class_Distribution.py
import streamlit as st
import plotly.express as px
from utils import load_sample_csv, ensure_feature_order

st.set_page_config(layout="wide")
st.title("📚 Class Distribution (sample)")

try:
    df = load_sample_csv()
except FileNotFoundError:
    st.error("sample_creditcard.csv missing.")
    st.stop()

df = ensure_feature_order(df)

st.markdown("Distribution of the `Class` column in the sample dataset.")
fig = px.pie(df, names="Class", hole=0.45, title="Class share (0 = genuine, 1 = fraud)")
st.plotly_chart(fig, use_container_width=True)

st.markdown("Show counts by class:")
st.write(df["Class"].value_counts())

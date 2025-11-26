import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from utils import load_sample_csv, download_link_html, ensure_feature_order

st.set_page_config(page_title="Credit Card Fraud — Ultra Pro", layout="wide")

st.title("💳 Credit Card Fraud Detection — Ultra Pro Max")
st.write("Single-page mode enabled. Pages removed.")

try:
    df = load_sample_csv()
    df = ensure_feature_order(df)
    st.success("sample_creditcard.csv loaded successfully.")
except Exception as e:
    st.error("sample_creditcard.csv missing or unreadable.")
    st.stop()

st.write("### Preview Data")
st.dataframe(df.head(), use_container_width=True)

st.markdown(download_link_html(df), unsafe_allow_html=True)

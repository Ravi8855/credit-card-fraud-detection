import streamlit as st
import plotly.express as px
from utils import load_sample_csv


st.set_page_config(page_title="Class Distribution", layout="wide")
st.markdown("# 📈 Class Distribution (sample)")


try:
df = load_sample_csv()
except FileNotFoundError:
st.error("sample_creditcard.csv not found.")
st.stop()


fig = px.histogram(df, x='Class', labels={'Class': 'Class (0=Not Fraud,1=Fraud)'}, title='Class distribution (sample)')
st.plotly_chart(fig, use_container_width=True)


st.markdown('## Quick stats')
st.write(df['Class'].value_counts())
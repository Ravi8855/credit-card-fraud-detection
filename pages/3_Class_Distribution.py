import streamlit as st
import plotly.express as px
from utils import load_sample_csv

st.title("📊 Class Distribution")

df = load_sample_csv()

fig = px.histogram(df, x="Class", title="Class Distribution (0 = Genuine, 1 = Fraud)")
st.plotly_chart(fig)

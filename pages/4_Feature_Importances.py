import streamlit as st
import plotly.express as px
from utils import load_sample_csv, train_model_pipeline

st.title("🔎 Top Feature Importances")

df = load_sample_csv()
metrics = train_model_pipeline(df)

feat = metrics["feat_imp"].head(12).sort_values("abs_coef", ascending=True)

fig = px.bar(
    feat,
    x="abs_coef",
    y="feature",
    orientation="h",
    title="Top Influential Features",
    labels={"abs_coef": "Absolute Coefficient Value"},
)

st.plotly_chart(fig)

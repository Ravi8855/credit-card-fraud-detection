import streamlit as st
import plotly.express as px
from utils import load_sample_csv, train_model_pipeline


st.set_page_config(page_title="Feature Importances", layout="wide")
st.markdown("# 🔎 Top Feature Importances (by |coef|)")


try:
df = load_sample_csv()
except FileNotFoundError:
st.error("sample_creditcard.csv not found.")
st.stop()


metrics = train_model_pipeline(df)
fi = metrics['feat_imp'].head(20)
fig = px.bar(fi, x='abs_coef', y='feature', orientation='h', labels={'abs_coef':'abs(coef)'}, title='Top features')
st.plotly_chart(fig, use_container_width=True)


st.dataframe(fi)
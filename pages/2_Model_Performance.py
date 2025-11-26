import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from utils import load_sample_csv, train_model_pipeline

st.title("📈 Model Performance")

df = load_sample_csv()
metrics = train_model_pipeline(df)

st.subheader("Confusion Matrix")
cm = metrics["confusion_matrix"]

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white")
st.pyplot(fig)

st.subheader("Metrics")
st.text(metrics["classification_report"])

st.subheader("ROC Curve")
roc_fig = px.area(
    x=metrics["roc_fpr"],
    y=metrics["roc_tpr"],
    title=f"ROC Curve — AUC = {metrics['roc_auc']:.4f}",
)
st.plotly_chart(roc_fig)

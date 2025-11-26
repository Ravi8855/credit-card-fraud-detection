# pages/2_Model_Performance.py
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from utils import load_sample_csv, train_model_pipeline

st.set_page_config(page_title="Model Performance", layout="wide")
st.markdown("## 🧠 Model Performance & Visualizations")

try:
    df = load_sample_csv()
except FileNotFoundError:
    st.error("sample_creditcard.csv missing.")
    st.stop()

# train (cached)
metrics = train_model_pipeline(df)

# Confusion matrix (matplotlib)
cm = metrics["confusion_matrix"]
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white", fontsize=14)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig, use_container_width=True)

# metrics
st.markdown("### Metrics")
st.write(f"**Accuracy:** `{metrics['accuracy']:.4f}`")
st.text(metrics["classification_report"])

# ROC
st.markdown("### ROC Curve")
roc_fig = px.area(
    x=metrics["roc_fpr"],
    y=metrics["roc_tpr"],
    title=f"ROC Curve (AUC = {metrics['roc_auc']:.4f})",
    labels={"x": "FPR", "y": "TPR"},
)
roc_fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash"))
st.plotly_chart(roc_fig, use_container_width=True)

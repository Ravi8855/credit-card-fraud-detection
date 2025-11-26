import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
from datetime import datetime

from utils import (
    load_sample_csv,
    download_link_html,
    ensure_feature_order,
    train_model_pipeline,
    save_trained_model,
)

# Streamlit page config
st.set_page_config(
    page_title="Credit Card Fraud — Ultra Pro Max",
    layout="wide",
    page_icon="💳"
)

# Styling
st.markdown("""
<style>
body { background:#070b10; color:#e6eef6; }
.stApp { background: linear-gradient(180deg,#07101a 0%, #04121a 100%); }
.card { background:rgba(255,255,255,0.03); padding:18px; border-radius:12px; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 💳 Credit Card Fraud Detection — Ultra Pro Max Dashboard")
st.write("All analytics + training + prediction on ONE page.")

# Load dataset
try:
    df = load_sample_csv()
except:
    st.error("sample_creditcard.csv missing.")
    st.stop()

# ---------- Dataset Overview ----------
st.markdown("### 📘 Dataset Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{df.shape[0]:,}")
c2.metric("Features", df.shape[1] - 1)
c3.metric("Fraud %", f"{df['Class'].mean()*100:.3f}%")

if st.checkbox("Show sample data"):
    st.dataframe(df.head(100), use_container_width=True)

# ---------- Train Model ----------
st.markdown("### 🧠 Model Training")

if st.button("Train Model (Logistic Regression)"):
    with st.spinner("Training model..."):
        metrics = train_model_pipeline(df)
        save_trained_model(metrics["model"])
    st.success("Model trained successfully!")

    st.write("**Accuracy:**", metrics["accuracy"])
    st.text(metrics["classification_report"])

    # Confusion Matrix
    cm = metrics["confusion_matrix"]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white")
    st.pyplot(fig)

    # ROC Curve
    roc_fig = px.line(
        x=metrics["roc_fpr"],
        y=metrics["roc_tpr"],
        labels={"x": "FPR", "y": "TPR"},
        title=f"ROC Curve (AUC={metrics['roc_auc']:.4f})"
    )
    st.plotly_chart(roc_fig, use_container_width=True)

# ---------- Manual Prediction ----------
st.markdown("### 🔮 Manual Prediction")

loaded = None
try:
    loaded = joblib.load("trained_model.pkl")
    st.success("Model loaded.")
except:
    st.warning("Train model first.")

if loaded:
    feature_inputs = {}
    cols = st.columns(4)
    for i, col in enumerate(loaded.feature_names_in_):
        with cols[i % 4]:
            feature_inputs[col] = st.number_input(col, value=0.0)

    if st.button("Predict Now"):
        x = np.array([list(feature_inputs.values())])
        pred = loaded.predict(x)[0]
        st.success("Fraud ❗" if pred == 1 else "Legit ✅")

# ---------- Download ----------
st.markdown("### 📥 Download Dataset")
st.markdown(download_link_html(df), unsafe_allow_html=True)

st.caption("Made with ❤️ — Ultra Pro Max Edition")

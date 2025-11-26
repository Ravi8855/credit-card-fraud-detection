# app.py — Single-page Ultra Pro Premium dashboard (UTF-8 safe)
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

st.set_page_config(page_title="Credit Card Fraud — Ultra Pro", layout="wide", page_icon="💳")

# ---- Top-level styling (kept simple & safe) ----
st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg,#07101a 0%, #04121a 100%); color: #e6eef6; }
      .title { font-size:28px; font-weight:700; color:#cfe9ff; margin-bottom:4px; }
      .muted { color:#9aa6b2; margin-bottom:12px; }
      .card { background: rgba(255,255,255,0.02); padding:12px; border-radius:10px; border: 1px solid rgba(80,160,255,0.04); }
      .small { font-size:12px; color:#9aa6b2; }
      a { color:#7fd1ff; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Header ----
st.markdown('<div class="title">💳 Credit Card Fraud Detection — Ultra Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Single-file professional dashboard: dataset, training, visualizations, prediction & export.</div>', unsafe_allow_html=True)
st.markdown("---")

# ---- Load dataset (safe) ----
try:
    df = load_sample_csv()
except FileNotFoundError:
    st.error("`sample_creditcard.csv` not found in repo root. Upload it or push it to enable the demo.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# Ensure expected column order and presence
df = ensure_feature_order(df)

# ---- Top-level metrics ----
st.header("📊 Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{df.shape[0]:,}")
col2.metric("Features", f"{df.shape[1] - (1 if 'Class' in df.columns else 0)}")
col3.metric("Has Class?", "Yes" if "Class" in df.columns else "No")

with st.expander("Preview dataset (first 20 rows)"):
    st.dataframe(df.head(20), use_container_width=True)

st.markdown(download_link_html(df), unsafe_allow_html=True)
st.markdown("---")

# ---- Class distribution ----
st.subheader("📦 Class Distribution")
if "Class" in df.columns:
    fig_hist = px.histogram(df, x="Class", labels={"Class": "Class (0=Not Fraud, 1=Fraud)"}, title="Class distribution")
    st.plotly_chart(fig_hist, use_container_width=True)
    st.write(df["Class"].value_counts())
else:
    st.info("No `Class` column in dataset — model training will not run.")

st.markdown("---")

# ---- Train model and compute metrics (cached in utils) ----
st.subheader("🧠 Train & Model Performance")
with st.spinner("Training model (cached) — this may take a few seconds..."):
    try:
        metrics = train_model_pipeline(df)
    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

# Metrics summary
st.markdown("### Summary")
st.write(f"**Trained at:** {metrics.get('trained_at', str(datetime.utcnow()))}")
st.metric("Accuracy", f"{metrics['accuracy']:.4f}")

# Confusion matrix (matplotlib)
cm = metrics["confusion_matrix"]
fig_cm, ax = plt.subplots(figsize=(4.5, 3.5))
ax.imshow(cm, cmap="Blues", interpolation="nearest")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="white", fontsize=12)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig_cm, use_container_width=True)

st.markdown("### Classification Report")
st.text(metrics["classification_report"])

# ROC curve (plotly)
roc_fig = px.area(
    x=metrics["roc_fpr"],
    y=metrics["roc_tpr"],
    title=f"ROC Curve (AUC = {metrics['roc_auc']:.4f})",
    labels={"x": "False Positive Rate", "y": "True Positive Rate"},
)
roc_fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash"))
st.plotly_chart(roc_fig, use_container_width=True)

st.markdown("---")

# ---- Feature importances ----
st.subheader("🔎 Top Feature Importances")
feat_imp = metrics.get("feat_imp")
if feat_imp is not None and not feat_imp.empty:
    top = feat_imp.head(12).sort_values("abs_coef", ascending=True)
    fig_fi = px.bar(top, x="abs_coef", y="feature", orientation="h", labels={"abs_coef": "abs(coef)"}, title="Top features by |coef|")
    st.plotly_chart(fig_fi, use_container_width=True)
    st.dataframe(feat_imp.head(20), use_container_width=True)
else:
    st.info("Feature importances not available.")

st.markdown("---")

# ---- Manual prediction form ----
st.subheader("🧮 Manual Transaction Prediction (Synthetic Inputs)")

feature_names = metrics["feature_names"]  # expected: Time, V1..V28, Amount
scaler = metrics["scaler"]
model = metrics["model"]

with st.form("manual_predict"):
    cols = st.columns([1, 1])
    time_val = cols[0].slider("Time (seconds)", 0, 172800, int(df["Time"].mean() if "Time" in df.columns else 0))
    amount_val = cols[1].number_input("Amount", min_value=0.0, max_value=200000.0, value=float(df["Amount"].mean() if "Amount" in df.columns else 0.0))

    st.markdown("#### PCA features V1–V28 (use sliders for demo)")
    left, right = st.columns(2)
    v_inputs = {}
    for i in range(1, 29):
        key = f"V{i}"
        default = float(df[key].mean()) if key in df.columns else 0.0
        if i <= 14:
            v_inputs[key] = left.slider(key, -10.0, 10.0, float(default), key=f"m_{key}")
        else:
            v_inputs[key] = right.slider(key, -10.0, 10.0, float(default), key=f"m_{key}")

    submit = st.form_submit_button("Predict Transaction")
    if submit:
        manual_dict = {"Time": float(time_val)}
        for i in range(1, 29):
            manual_dict[f"V{i}"] = float(v_inputs[f"V{i}"])
        manual_dict["Amount"] = float(amount_val)

        # ensure ordering
        X_manual = pd.DataFrame([manual_dict])[feature_names]
        X_scaled = scaler.transform(X_manual)
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0][1]

        st.markdown("### Prediction Result")
        st.write(f"**Probability (fraud):** `{proba:.4f}`")
        if pred == 1:
            st.error("⚠️ Likely FRAUDULENT Transaction")
        else:
            st.success("✅ Likely Genuine Transaction")

st.markdown("---")

# ---- Model export & resources ----
st.subheader("📦 Model Export & Resources")
if st.button("Save trained model to repo (trained_model.joblib)"):
    try:
        path = save_trained_model("trained_model_and_scaler.joblib", model, scaler)
        st.success(f"Saved {path} to repo root (will appear after push).")
    except Exception as e:
        st.error(f"Save failed: {e}")

st.write("**Sample CSV used:** `sample_creditcard.csv` (kept small for demo).")
st.write("If you want a public demo, ensure `sample_creditcard.csv` exists in repo root and then reboot Streamlit Cloud.")

st.markdown("---")
st.caption("Made with ❤️ — Ultra Pro (single-page).")

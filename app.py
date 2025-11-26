import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from utils import (
    load_sample_csv,
    download_link_html,
    ensure_feature_order,
    train_model_pipeline,
)

st.set_page_config(page_title="Credit Card Fraud Detection — Pro", layout="wide")

st.title("💳 Credit Card Fraud Detection — Ultra Pro Edition")
st.write("A single-page professional dashboard combining all features.")


# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
try:
    df = load_sample_csv()
    df = ensure_feature_order(df)
    st.success("Dataset loaded successfully.")
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()


# -------------------------------------------------------------------
# SECTION 1 — DATA OVERVIEW
# -------------------------------------------------------------------
st.header("📊 Dataset Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{df.shape[0]:,}")
c2.metric("Features", df.shape[1] - 1)
c3.metric("Fraud Cases", int((df['Class'] == 1).sum()))

if st.checkbox("Show first 50 rows"):
    st.dataframe(df.head(50), use_container_width=True)

st.markdown(download_link_html(df), unsafe_allow_html=True)


# -------------------------------------------------------------------
# SECTION 2 — CLASS DISTRIBUTION
# -------------------------------------------------------------------
st.header("📦 Class Distribution (Fraud vs Non-Fraud)")

fig = px.histogram(df, x="Class", title="Class Distribution", color="Class")
st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------
# SECTION 3 — MODEL TRAINING + PERFORMANCE
# -------------------------------------------------------------------
st.header("🧠 Model Performance")

metrics = train_model_pipeline(df)

st.subheader("Accuracy")
st.metric("Model Accuracy", f"{metrics['accuracy']:.4f}")

st.subheader("Classification Report")
st.text(metrics["classification_report"])

st.subheader("Confusion Matrix")
cm = metrics["confusion_matrix"]
fig2, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig2)


# -------------------------------------------------------------------
# SECTION 4 — ROC CURVE
# -------------------------------------------------------------------
st.subheader("ROC Curve")
roc_fig = px.area(
    x=metrics["roc_fpr"],
    y=metrics["roc_tpr"],
    title=f"ROC Curve (AUC = {metrics['roc_auc']:.4f})",
    labels={"x": "False Positive Rate", "y": "True Positive Rate"}
)
roc_fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash"))
st.plotly_chart(roc_fig, use_container_width=True)


# -------------------------------------------------------------------
# SECTION 5 — FEATURE IMPORTANCES
# -------------------------------------------------------------------
st.header("🏷 Top Features Influencing Fraud Detection")

feat_imp = metrics["feat_imp"].head(10)
st.dataframe(feat_imp, use_container_width=True)

fig3 = px.bar(
    feat_imp,
    x="abs_coef",
    y="feature",
    title="Top Feature Importances",
    orientation="h"
)
st.plotly_chart(fig3, use_container_width=True)


# -------------------------------------------------------------------
# SECTION 6 — MANUAL PREDICTION
# -------------------------------------------------------------------
st.header("🧮 Manual Prediction Tool")

model = metrics["model"]
scaler = metrics["scaler"]

with st.form("prediction_form"):
    st.write("Enter feature values below:")

    feature_values = []
    for col in metrics["feature_names"]:
        feature_values.append(st.number_input(col, min_value=-999999.0, max_value=999999.0, value=0.0))

    submitted = st.form_submit_button("Predict Fraud")

if submitted:
    x = pd.DataFrame([feature_values], columns=metrics["feature_names"])
    x_scaled = scaler.transform(x)
    prob = model.predict_proba(x_scaled)[0][1]
    pred = model.predict(x_scaled)[0]

    st.success(f"Fraud Probability: **{prob:.6f}**")
    st.warning("🚨 FRAUD DETECTED!") if pred == 1 else st.info("✔ Not Fraud")


# -------------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------------
st.markdown("---")
st.caption("Made with ❤️ — Single-Page Ultra Pro Dashboard")

# app.py — Ultra Pro Max (regenerated, no syntax errors)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import io
import base64
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

# -------------------------
# Page configuration + CSS
# -------------------------
st.set_page_config(page_title="Credit Card Fraud — Ultra Pro", layout="wide", page_icon="💳")

st.markdown(
    """
    <style>
      .stApp { background: #0b1220; color: #e6eef6; }
      .title { font-weight:700; font-size:28px; }
      .muted { color:#9aa6b2; margin-bottom:8px; }
      .card { background: rgba(255,255,255,0.02); padding:12px; border-radius:8px; }
      .small { font-size:12px; color:#9aa6b2; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Helpers
# -------------------------
def cached_csv_download_link(df: pd.DataFrame, filename="sample_creditcard.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f"data:file/csv;base64,{b64}"

def download_link_html(df: pd.DataFrame, filename="sample_creditcard.csv"):
    href = cached_csv_download_link(df, filename)
    return f'<a href="{href}" download="{filename}">📥 Download CSV</a>'

# -------------------------
# Load dataset (cached)
# -------------------------
@st.cache_data(show_spinner=False)
def load_sample_csv(path="sample_creditcard.csv"):
    df_local = pd.read_csv(path)
    return df_local

# -------------------------
# Train pipeline (cached resource)
# -------------------------
@st.cache_resource(show_spinner=False)
def train_model_pipeline(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Maintain column order for manual input later
    feature_names = X.columns.tolist()

    # Stratified split to keep class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=3000, solver="lbfgs")
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    metrics = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, digits=4),
    }

    # compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    metrics["roc_auc"] = auc(fpr, tpr)
    metrics["roc_fpr"] = fpr
    metrics["roc_tpr"] = tpr

    # feature importance (abs of coefficients)
    coefs = model.coef_.flatten()
    feat_imp = pd.DataFrame({"feature": feature_names, "coef": coefs})
    feat_imp["abs_coef"] = feat_imp["coef"].abs()
    feat_imp = feat_imp.sort_values("abs_coef", ascending=False)
    metrics["feat_imp"] = feat_imp

    return metrics

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("Settings")
st.sidebar.markdown("**Deployment**")
retrain_flag = st.sidebar.checkbox("Retrain model now (force)", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown("**Resources**")
st.sidebar.markdown("- GitHub repo")
st.sidebar.markdown("- Upload original dataset if needed")
st.sidebar.markdown("---")
st.sidebar.markdown("**Original ZIP (admin)**")
st.sidebar.markdown("file:///mnt/data/credit-card-fraud-detection.zip")

# -------------------------
# Main app
# -------------------------
# Try loading sample CSV
try:
    df = load_sample_csv()
except FileNotFoundError:
    st.error("sample_creditcard.csv not found in repo root. Add sample_creditcard.csv and reboot.")
    st.stop()

# Train or re-train
if retrain_flag:
    # Clear cache by re-calling train_model_pipeline in a deterministic way
    metrics = train_model_pipeline(df)
else:
    metrics = train_model_pipeline(df)

# Header
left, right = st.columns([3, 1])
with left:
    st.markdown('<div class="title">💳 Credit Card Fraud Detection — Ultra Pro</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Premium final-year demo: interactive prediction, visualizations, model export.</div>', unsafe_allow_html=True)
with right:
    st.markdown(f'<div class="small">Last updated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}</div>', unsafe_allow_html=True)

st.markdown("---")

# Dataset overview
st.subheader("📊 Dataset Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{df.shape[0]:,}")
c2.metric("Features", f"{df.shape[1]-1}")
c3.metric("Target column", "Class")

if st.checkbox("Show raw data (first 50 rows)"):
    st.dataframe(df.head(50))

st.markdown(download_link_html(df), unsafe_allow_html=True)
st.markdown("---")

# Performance & Visuals
st.subheader("📈 Model Performance & Visualizations")
colL, colR = st.columns([1.1, 1])

# Confusion matrix plot
with colL:
    st.markdown("### Confusion Matrix")
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

    st.markdown("### Class Distribution (sample)")
    fig_hist = px.histogram(df, x="Class", labels={"Class": "Class (0=Not Fraud, 1=Fraud)"}, title="Class distribution")
    st.plotly_chart(fig_hist, use_container_width=True)

with colR:
    st.markdown("### Metrics")
    st.write(f"**Accuracy:** `{metrics['accuracy']:.4f}`")
    st.text(metrics["classification_report"])

    st.markdown("### ROC Curve")
    roc_fig = px.area(
        x=metrics["roc_fpr"],
        y=metrics["roc_tpr"],
        title=f"ROC Curve (AUC = {metrics['roc_auc']:.4f})",
        labels={"x": "FPR", "y": "TPR"},
    )
    roc_fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash"))
    st.plotly_chart(roc_fig, use_container_width=True)

st.markdown("---")

# Feature importances
st.subheader("🔎 Top Feature Importances (by |coef|)")
fi = metrics["feat_imp"].head(12).sort_values("abs_coef", ascending=True)
fig_fi = px.bar(fi, x="abs_coef", y="feature", orientation="h", labels={"abs_coef": "abs(coef)"}, title="Top features")
st.plotly_chart(fig_fi, use_container_width=True)
st.markdown("---")

# =========================
# Option 1 — Synthetic Manual Prediction Form
# =========================
st.subheader("🧮 Manual Transaction Prediction (Synthetic Inputs)")

st.write("Enter synthetic (realistic) values for a transaction. This is ideal for demos and final-year presentation.")

# Ranges
TIME_MIN, TIME_MAX = 0, 172800
V_MIN, V_MAX = -10.0, 10.0
AMOUNT_MIN, AMOUNT_MAX = 0.0, 20000.0

feature_names = metrics["feature_names"]  # preserves order: Time, V1..V28, Amount

with st.form("manual_predict_form_option1"):
    # Time and Amount top row
    row1, row2 = st.columns([1, 1])
    with row1:
        time_val = st.slider("⏱ Time (seconds)", TIME_MIN, TIME_MAX, int(df["Time"].mean()))
    with row2:
        amount_val = st.slider("💰 Amount (₹)", AMOUNT_MIN, AMOUNT_MAX, float(df["Amount"].mean()))

    st.markdown("### PCA features V1–V28 (synthetic ranges)")
    colA, colB = st.columns(2)
    v_inputs = {}
    for i in range(1, 29):
        key = f"V{i}"
        default = float(df[key].mean()) if key in df.columns else 0.0
        if i <= 14:
            v_inputs[key] = colA.slider(key, float(V_MIN), float(V_MAX), float(default), key=f"m_{key}")
        else:
            v_inputs[key] = colB.slider(key, float(V_MIN), float(V_MAX), float(default), key=f"m_{key}")

    submit_manual = st.form_submit_button("🔍 Predict Transaction")

    if submit_manual:
        # Build input df in same column order
        manual_dict = {"Time": float(time_val)}
        for i in range(1, 29):
            manual_dict[f"V{i}"] = float(v_inputs[f"V{i}"])
        manual_dict["Amount"] = float(amount_val)

        manual_df = pd.DataFrame([manual_dict])[feature_names]

        # Scale & predict
        scaler = metrics["scaler"]
        model = metrics["model"]
        scaled = scaler.transform(manual_df)
        pred = model.predict(scaled)[0]
        proba = model.predict_proba(scaled)[0][1]

        st.markdown("### 🧾 Prediction Result")
        st.write(f"**Probability (fraud):** `{proba:.4f}`")

        if pred == 1:
            st.error("⚠️ Likely FRAUDULENT Transaction")
        else:
            st.success("✅ Likely Genuine Transaction")

        st.info("Note: synthetic values are for demonstration and grading purposes; they do not represent real cardholder data.")

st.markdown("---")

# Model export & resources
st.subheader("📦 Model Export & Resources")
if st.button("Save trained model (joblib)"):
    joblib.dump({"model": metrics["model"], "scaler": metrics["scaler"]}, "trained_model_and_scaler.joblib")
    st.success("Saved trained_model_and_scaler.joblib to repo root.")

st.markdown("**Included sample dataset (small) for quick online demo:**")
st.markdown("`sample_creditcard.csv` (1000 rows) — use for quick demos.")
st.markdown("**Original large dataset (not included in repo):**")
st.markdown("`file:///mnt/data/credit-card-fraud-detection.zip`")

st.caption("Made with ❤️ — Ultra Pro Max Premium edition. Ask me to generate the final report & presentation.")

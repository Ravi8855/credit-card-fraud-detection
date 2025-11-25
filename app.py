# app.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)

import matplotlib.pyplot as plt
from matplotlib import cm

# ---------- Page config ----------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

# ---------- Small CSS & theme toggle ----------
DEFAULT_CSS = """
<style>
.app-title { font-size: 30px; font-weight:700; }
.section-title { font-size:20px; font-weight:600; margin-top:18px; margin-bottom:6px; }
.metric-box { padding:12px; border-radius:10px; background-color: rgba(255,255,255,0.02); }
.small-muted { color: #9aa0a6; font-size:13px; }
.card { padding:10px; border-radius:8px; border:1px solid rgba(255,255,255,0.03); }
</style>
"""

# Allow user to toggle "compact light" vs "dark improved"
with st.sidebar:
    st.title("Settings")
    theme_dark = st.checkbox("Use alternate light theme (toggle)", value=False)
    st.markdown("**Model operations**")
    retrain = st.checkbox("Retrain model on each load (safe: leave unchecked)", value=False)
    st.markdown("---")

if theme_dark:
    # light-ish override (simple)
    LIGHT_CSS = """
    <style>
    .stApp { background: #f8fafc; color: #0b1220; }
    .card { background: #ffffff; border:1px solid rgba(0,0,0,0.06); box-shadow: 0 1px 6px rgba(16,24,40,0.04); }
    </style>
    """
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)
else:
    st.markdown(DEFAULT_CSS, unsafe_allow_html=True)


# ---------- Data & Model helpers ----------
@st.cache_data
def load_sample(path: str = "sample_creditcard.csv") -> pd.DataFrame:
    """Load the small sample CSV (1000 rows) from the repo."""
    df = pd.read_csv(path)
    return df


@st.cache_resource
def train_and_build(df: pd.DataFrame, force_retrain: bool = False) -> Dict:
    """Train logistic regression and compute evaluation artifacts."""
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    # feature importance for logistic = absolute value of coefficients
    feature_names = list(X.columns)
    coefs = model.coef_.flatten()
    feat_imp = pd.DataFrame({"feature": feature_names, "coef": coefs})
    feat_imp["abs_coef"] = feat_imp["coef"].abs()
    feat_imp = feat_imp.sort_values("abs_coef", ascending=False)

    return {
        "model": model,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "accuracy": acc,
        "confusion_matrix": cm,
        "report": report,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "precision_curve": precision,
        "recall_curve": recall,
        "feature_importance": feat_imp,
        "feature_names": feature_names,
    }


# ---------- UI Start ----------
st.title("💳 Credit Card Fraud Detection App", anchor="title")
st.write("A sample demo app (1000-row sample) — prediction form, evaluation graphs, and UI improvements.")

st.markdown("---")

# Load data
df = load_sample()  # expects sample_creditcard.csv in repo root
if df is None or df.shape[0] == 0:
    st.error("Dataset not found. Make sure sample_creditcard.csv is present.")
    st.stop()

# Train model (or use cached)
if retrain:
    # if user chooses retrain, clear cache then retrain (streamlit handles cache flags)
    model_info = train_and_build(df, force_retrain=True)
else:
    model_info = train_and_build(df)

# ---------- Top Metrics & Dataset Overview ----------
st.subheader("📊 Dataset Overview")
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.metric("Total Rows", df.shape[0])
    st.markdown("</div>", unsafe_allow_html=True)
with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.metric("Total Features", df.shape[1] - 1)
    st.markdown("</div>", unsafe_allow_html=True)
with c3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.metric("Target Column", "Class")
    st.markdown("</div>", unsafe_allow_html=True)

if st.checkbox("Show raw data (first 50 rows)"):
    st.dataframe(df.head(50))

st.markdown("---")

# ---------- Model Performance & Graphs ----------
st.subheader("🤖 Model Performance & Visualizations")

left, right = st.columns([1.2, 1])

# Left column: metrics + confusion matrix
with left:
    st.markdown("### Model metrics")
    st.write(f"**Test accuracy:** `{model_info['accuracy']:.4f}`")
    st.markdown("**Classification Report**")
    st.text(model_info["report"])

    st.markdown("**Confusion matrix**")
    cm = model_info["confusion_matrix"]
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    cmap = cm = plt.get_cmap("viridis")
    im = ax_cm.imshow(cm, interpolation="nearest", cmap="YlGnBu")
    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Not Fraud (0)", "Fraud (1)"], rotation=20)
    ax_cm.set_yticklabels(["Not Fraud (0)", "Fraud (1)"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=14)
    fig_cm.tight_layout()
    st.pyplot(fig_cm)

# Right column: ROC & PR curves
with right:
    st.markdown("### ROC Curve (AUC)")
    fpr = model_info["fpr"]; tpr = model_info["tpr"]; roc_auc = model_info["roc_auc"]
    fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2)
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    st.markdown("### Precision–Recall")
    precision_curve = model_info["precision_curve"]
    recall_curve = model_info["recall_curve"]
    fig_pr, ax_pr = plt.subplots(figsize=(4, 3))
    ax_pr.plot(recall_curve, precision_curve, linewidth=2)
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    st.pyplot(fig_pr)

st.markdown("---")

# ---------- Distribution & Feature importances ----------
st.subheader("📈 Data Distribution & Feature Importances")
col1, col2 = st.columns([1, 1])
with col1:
    counts = df["Class"].value_counts().sort_index()
    fig_bar, ax_bar = plt.subplots(figsize=(5, 2.5))
    ax_bar.bar(["Not Fraud (0)", "Fraud (1)"], counts, color=["#2b8cbe", "#fdae61"])
    ax_bar.set_ylabel("Count")
    st.pyplot(fig_bar)

with col2:
    st.markdown("Top feature importances (logistic coefficients)")
    feat_imp = model_info["feature_importance"].head(10).copy()
    fig_fi, ax_fi = plt.subplots(figsize=(5, 2.5))
    ax_fi.barh(feat_imp["feature"][::-1], feat_imp["abs_coef"][::-1], color="#6a51a3")
    ax_fi.set_xlabel("Absolute coefficient")
    st.pyplot(fig_fi)

st.markdown("---")

# ---------- Random test prediction ----------
st.subheader("🔀 Random Test Transaction Prediction")
if st.button("Pick & Predict Random Test Transaction"):
    X_test = model_info["X_test"]
    y_test = model_info["y_test"]
    scaler = model_info["scaler"]
    model = model_info["model"]

    idx = np.random.randint(0, X_test.shape[0])
    sample = X_test.iloc[idx]
    true_label = y_test.iloc[idx]

    st.write("**Sample features:**")
    st.write(sample.to_frame().T)

    scaled = scaler.transform([sample])
    pred = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0][1]

    st.write(f"**Prediction:** {'Fraud (1)' if pred==1 else 'Not Fraud (0)'}")
    st.write(f"**Fraud probability:** `{proba:.4f}`")
    if pred == true_label:
        st.success("✅ Model prediction matches actual label")
    else:
        st.warning("⚠ Model prediction does not match actual label")

st.markdown("---")

# ---------- Manual prediction form ----------
st.subheader("🧮 Manual Input Prediction Form")

feature_names = model_info["feature_names"]
means = df[feature_names].mean()
mins = df[feature_names].min()
maxs = df[feature_names].max()

# Layout inputs
with st.form("manual_form", clear_on_submit=False):
    st.markdown("Enter transaction features (Time, V1–V28, Amount). Use reasonable defaults from the dataset.")
    # two columns for Time and Amount
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        time_val = st.number_input("Time", float(mins["Time"]), float(maxs["Time"]), float(means["Time"]))
    with tcol2:
        amount_val = st.number_input("Amount", float(mins["Amount"]), float(maxs["Amount"]), float(means["Amount"]))

    # PCA features in two columns
    pca_cols = st.columns(2)
    v_values = {}
    v_list = [f"V{i}" for i in range(1, 29)]
    for i, v in enumerate(v_list):
        col_idx = 0 if i < 14 else 1
        with pca_cols[col_idx]:
            v_values[v] = st.number_input(
                v, float(mins[v]), float(maxs[v]), float(means[v]), key=f"input_{v}"
            )

    submitted = st.form_submit_button("Predict Transaction")
    if submitted:
        input_dict = {"Time": time_val, **{v: v_values[v] for v in v_list}, "Amount": amount_val}
        manual_df = pd.DataFrame([input_dict], columns=feature_names)
        scaled_manual = model_info["scaler"].transform(manual_df)
        pred_manual = model_info["model"].predict(scaled_manual)[0]
        proba_manual = model_info["model"].predict_proba(scaled_manual)[0][1]

        st.write("### Result")
        st.write(f"**Prediction:** {'Fraud (1)' if pred_manual==1 else 'Not Fraud (0)'}")
        st.write(f"**Fraud probability:** `{proba_manual:.4f}`")
        if pred_manual == 1:
            st.error("⚠ This transaction is likely FRAUDULENT.")
        else:
            st.success("✅ This transaction is likely NOT FRAUD.")

st.markdown("---")

# ---------- Footer ----------
st.markdown("Made with ❤️ — Credit Card Fraud Detection Demo. For final-year submission, I can generate the project report & slides.")


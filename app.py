import streamlit as st
import pandas as pd
import numpy as np

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

import matplotlib.pyplot as plt


# ---------- DATA & MODEL ----------

@st.cache_data
def load_data():
    # Sample dataset (1000 rows) for fast online performance
    df = pd.read_csv("sample_creditcard.csv")
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Keep X_test, y_test (unscaled) for display & form defaults
    return {
        "model": model,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "accuracy": acc,
        "cm": cm,
        "report": report_text,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "feature_names": list(X.columns),
    }


# ---------- PAGE CONFIG & BASIC STYLING ----------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide",
    page_icon="💳",
)

# Simple custom CSS for nicer spacing & cards
st.markdown(
    """
    <style>
    .main {
        padding: 1.5rem 3rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- TITLE ----------

st.title("💳 Credit Card Fraud Detection App")
st.write(
    "A Machine Learning model that detects fraudulent credit card transactions "
    "using a **sample dataset (1000 rows)** for fast online performance."
)

st.markdown("---")

# ---------- LOAD DATA ----------

df = load_data()
model_info = train_model(df)

# ---------- DATASET OVERVIEW SECTION ----------

st.subheader("📊 Dataset Overview")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Total Rows", df.shape[0])
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Total Features", df.shape[1] - 1)
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Fraud Column", "Class")
    st.markdown("</div>", unsafe_allow_html=True)

if st.checkbox("Show Raw Data (first 50 rows)"):
    st.dataframe(df.head(50))

st.markdown("---")

# ---------- MODEL PERFORMANCE & GRAPHS ----------

st.subheader("🤖 Model Performance & Visualizations")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.write(f"### ✅ Test Accuracy: `{model_info['accuracy']:.4f}`")

    st.write("#### Confusion Matrix")
    cm = model_info["cm"]

    fig_cm, ax_cm = plt.subplots()
    im = ax_cm.imshow(cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Not Fraud (0)", "Fraud (1)"])
    ax_cm.set_yticklabels(["Not Fraud (0)", "Fraud (1)"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig_cm)

with col_right:
    st.write("#### Classification Report")
    st.text(model_info["report"])

st.markdown("#### Fraud vs Non-Fraud Distribution")

col_bar1, col_bar2 = st.columns(2)
with col_bar1:
    counts = df["Class"].value_counts().sort_index()
    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(["Not Fraud (0)", "Fraud (1)"], counts)
    ax_bar.set_ylabel("Count")
    st.pyplot(fig_bar)

with col_bar2:
    st.write("Class Distribution (Normalized)")
    st.write(df["Class"].value_counts(normalize=True))

st.markdown("#### ROC Curve")

fig_roc, ax_roc = plt.subplots()
ax_roc.plot(model_info["fpr"], model_info["tpr"], label=f"AUC = {model_info['roc_auc']:.3f}")
ax_roc.plot([0, 1], [0, 1], linestyle="--")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend(loc="lower right")
st.pyplot(fig_roc)

st.markdown("---")

# ---------- RANDOM TRANSACTION PREDICTION ----------

st.subheader("🎲 Predict a Random Transaction from Test Set")

if st.button("Pick & Predict Random Transaction"):
    X_test = model_info["X_test"]
    y_test = model_info["y_test"]
    model = model_info["model"]
    scaler = model_info["scaler"]

    idx = np.random.randint(0, X_test.shape[0])
    row = X_test.iloc[idx]
    true_label = y_test.iloc[idx]

    st.write("**Transaction Features:**")
    st.write(row.to_frame().T)

    scaled = scaler.transform([row])
    pred = model.predict(scaled)[0]

    label_map = {0: "Not Fraud", 1: "Fraud"}

    st.write(f"### 🔵 Prediction: **{label_map[pred]}** (`{pred}`)")
    st.write(f"### 🟢 Actual: **{label_map[true_label]}** (`{true_label}`)")

    if pred == true_label:
        st.success("Model prediction is correct!")
    else:
        st.warning("Model prediction is incorrect.")

st.markdown("---")

# ---------- CUSTOM PREDICTION FORM (USER INPUT) ----------

st.subheader("🧮 Manual Input Prediction Form")

st.write(
    "Use this form to manually enter feature values (`Time`, `V1`–`V28`, `Amount`) "
    "and get a fraud / not-fraud prediction from the model."
)

feature_names = model_info["feature_names"]
model = model_info["model"]
scaler = model_info["scaler"]

# Use dataset statistics for reasonable default values
means = df[feature_names].mean()
mins = df[feature_names].min()
maxs = df[feature_names].max()

with st.form("manual_prediction_form"):
    st.markdown("#### Basic Features")

    c_time, c_amount = st.columns(2)
    with c_time:
        time_val = st.number_input(
            "Time",
            float(mins["Time"]),
            float(maxs["Time"]),
            float(means["Time"]),
        )
    with c_amount:
        amount_val = st.number_input(
            "Amount",
            float(mins["Amount"]),
            float(maxs["Amount"]),
            float(means["Amount"]),
        )

    st.markdown("#### PCA Features (V1 – V28)")
    # Show V1–V14 and V15–V28 in two columns for readability
    cols_pca = st.columns(2)
    v_values = {}

    v_list = [f"V{i}" for i in range(1, 29)]
    first_half = v_list[:14]
    second_half = v_list[14:]

    with cols_pca[0]:
        for v in first_half:
            v_values[v] = st.number_input(
                v,
                float(mins[v]),
                float(maxs[v]),
                float(means[v]),
            )
    with cols_pca[1]:
        for v in second_half:
            v_values[v] = st.number_input(
                v,
                float(mins[v]),
                float(maxs[v]),
                float(means[v]),
            )

    submitted = st.form_submit_button("🔍 Predict Manually")

    if submitted:
        # Build single-row DataFrame in correct column order
        input_dict = {
            "Time": time_val,
            **{v: v_values[v] for v in v_list},
            "Amount": amount_val,
        }

        manual_df = pd.DataFrame([input_dict], columns=feature_names)

        scaled_manual = scaler.transform(manual_df)
        pred_manual = model.predict(scaled_manual)[0]
        proba_manual = model.predict_proba(scaled_manual)[0][1]

        label_map = {0: "Not Fraud", 1: "Fraud"}

        st.write("### 🔎 Manual Prediction Result")
        st.write(f"**Prediction:** {label_map[pred_manual]} (`{pred_manual}`)")
        st.write(f"**Fraud Probability:** `{proba_manual:.4f}`")

        if pred_manual == 1:
            st.error("⚠ This transaction is likely FRAUDULENT.")
        else:
            st.success("✅ This transaction is likely NOT FRAUD.")

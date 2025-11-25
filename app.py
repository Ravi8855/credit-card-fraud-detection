import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# ================================
# 🎨 Streamlit Page Configuration
# ================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide",
    page_icon="🧾"
)

st.markdown("""
    <style>
    .main {background-color: #111;}
    .stApp {background-color: #111;}
    h1, h2, h3, h4, p, label, span {color: #e6e6e6 !important;}
    </style>
""", unsafe_allow_html=True)

# =======================================
# 📌 Load Dataset
# =======================================
@st.cache_data
def load_data():
    return pd.read_csv("sample_creditcard.csv")

df = load_data()

# =======================================
# 🎯 Train Model
# =======================================
X = df.drop("Class", axis=1)
y = df["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ===================================================
# 🧭 Sidebar Navigation
# ===================================================
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Visualizations", "🧮 Predict Fraud"]
)

# ===================================================
# 🏠 HOME PAGE
# ===================================================
if menu == "🏠 Home":
    st.title("🧾 Credit Card Fraud Detection App")
    st.write(
        "This machine-learning app detects fraudulent credit card transactions "
        "using a logistic regression model trained on a sample (1000 rows)."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Features", df.shape[1] - 1)
    col3.metric("Fraud Label", "Class")

    if st.checkbox("Show Raw Dataset (first 50 rows)"):
        st.dataframe(df.head(50))

# ===================================================
# 📊 VISUALIZATION PAGE
# ===================================================
elif menu == "📊 Visualizations":
    st.title("📊 Model Performance & Insights")

    # Confusion Matrix
    st.subheader("🔶 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # Classification Report
    st.subheader("📘 Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Fraud Distribution
    st.subheader("📌 Fraud Distribution")
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sns.countplot(x=df["Class"], palette="viridis", ax=ax2)
    st.pyplot(fig2)

    # ROC Curve
    st.subheader("📈 ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    graph_auc = auc(fpr, tpr)

    fig3, ax3 = plt.subplots(figsize=(5, 4))
    ax3.plot(fpr, tpr, label=f"AUC = {graph_auc:.4f}")
    ax3.plot([0, 1], [0, 1], linestyle="--")
    ax3.set_title("ROC Curve")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.legend()
    st.pyplot(fig3)

# ===================================================
# 🧮 PREDICT FRAUD PAGE
# ===================================================
elif menu == "🧮 Predict Fraud":
    st.title("🧮 Predict Fraud for a Single Transaction")

    st.write("Fill in the transaction fields below to check if it's fraudulent.")

    input_data = []

    # Input fields for V1-V28
    st.subheader("Transaction Features")
    cols = st.columns(3)
    feature_names = X.columns.tolist()

    for i, col_name in enumerate(feature_names):
        col = cols[i % 3]
        val = col.number_input(col_name, value=float(0), format="%.4f")
        input_data.append(val)

    # Convert to array
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    if st.button("🔍 Predict"):
        result = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if result == 1:
            st.error(f"⚠️ Fraudulent Transaction Detected! (Probability: {prob:.4f})")
        else:
            st.success(f"✅ Genuine Transaction (Probability: {prob:.4f})")

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------- Data & Model ----------

@st.cache_data
def load_data():
    df = pd.read_csv("sample_creditcard.csv")  # Using sample dataset for cloud deployment
    return df

@st.cache_resource
def train_model(df):
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
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return model, scaler, X_test, y_test, acc, cm, report

# ---------- Streamlit UI ----------

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection App")
st.write(
    """
A Machine Learning model that detects fraudulent credit card transactions  
using a **sample dataset** (1000 rows) for fast online performance.
"""
)

# Load data
df = load_data()

st.subheader("📊 Dataset Overview")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total Rows", df.shape[0])
with c2:
    st.metric("Total Features", df.shape[1] - 1) 
with c3:
    st.metric("Fraud Column", "Class")

if st.checkbox("Show Raw Data (first 50 rows)"):
    st.write(df.head(50))

st.markdown("---

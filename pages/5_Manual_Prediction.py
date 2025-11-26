import streamlit as st
import pandas as pd
from utils import load_sample_csv, train_model_pipeline

st.title("🧮 Manual Fraud Prediction")

df = load_sample_csv()
metrics = train_model_pipeline(df)

feature_names = metrics["feature_names"]

st.write("Enter synthetic (demo) values below:")

with st.form("manual_form"):
    time_val = st.number_input("Time", 0, 200000, 10000)
    amount_val = st.number_input("Amount", 0.0, 20000.0, 100.0)

    st.write("### PCA Components (V1–V28)")
    v_vals = {}
    for i in range(1, 29):
        v_vals[f"V{i}"] = st.slider(f"V{i}", -10.0, 10.0, 0.0)

    submit = st.form_submit_button("Predict")

if submit:
    new_row = {"Time": time_val}
    for i in range(1, 29):
        new_row[f"V{i}"] = v_vals[f"V{i}"]
    new_row["Amount"] = amount_val

    user_df = pd.DataFrame([new_row])[feature_names]

    scaled = metrics["scaler"].transform(user_df)
    pred = metrics["model"].predict(scaled)[0]
    proba = metrics["model"].predict_proba(scaled)[0][1]

    st.write(f"**Probability of Fraud:** `{proba:.4f}`")

    if pred == 1:
        st.error("⚠️ Fraudulent Transaction")
    else:
        st.success("✅ Genuine Transaction")

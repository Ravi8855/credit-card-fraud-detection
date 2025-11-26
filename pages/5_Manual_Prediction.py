# pages/5_Manual_Prediction.py
import streamlit as st
import pandas as pd
from utils import load_sample_csv, train_model_pipeline, ensure_feature_order

st.set_page_config(page_title="Manual Prediction", layout="wide")
st.markdown("## 🧾 Manual Transaction Prediction (Synthetic Inputs)")

try:
    df = load_sample_csv()
except FileNotFoundError:
    st.error("sample_creditcard.csv missing.")
    st.stop()

metrics = train_model_pipeline(df)
feature_names = metrics["feature_names"]

# ranges
TIME_MIN, TIME_MAX = 0, 172800
V_MIN, V_MAX = -10.0, 10.0
AMOUNT_MIN, AMOUNT_MAX = 0.0, 20000.0

with st.form("manual_predict_form"):
    col1, col2 = st.columns([1, 1])
    with col1:
        time_val = st.slider("⏱ Time (seconds)", TIME_MIN, TIME_MAX, int(df["Time"].mean()))
    with col2:
        amount_val = st.slider("💰 Amount", AMOUNT_MIN, AMOUNT_MAX, float(df["Amount"].mean()))

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

    submit = st.form_submit_button("🔍 Predict Transaction")
    if submit:
        manual_dict = {"Time": float(time_val)}
        for i in range(1, 29):
            manual_dict[f"V{i}"] = float(v_inputs[f"V{i}"])
        manual_dict["Amount"] = float(amount_val)

        manual_df = pd.DataFrame([manual_dict])[feature_names]
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
        st.info("Note: synthetic inputs are for demo only.")

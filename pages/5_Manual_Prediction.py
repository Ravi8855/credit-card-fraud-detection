# pages/5_Manual_Prediction.py
import streamlit as st
import pandas as pd
from utils import load_sample_csv, ensure_feature_order
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")
st.title("🧮 Manual Transaction Prediction (Synthetic Inputs)")

try:
    df = load_sample_csv()
except FileNotFoundError:
    st.error("sample_creditcard.csv missing.")
    st.stop()

df = ensure_feature_order(df)

# Train quick model (same pipeline as other pages)
X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
model = LogisticRegression(max_iter=2000, solver="lbfgs")
model.fit(X_train_s, y_train)

# UI
st.markdown("Enter synthetic values (demo ranges). Use sliders and press **Predict Transaction**.")

TIME_MIN, TIME_MAX = 0, 172800
V_MIN, V_MAX = -10.0, 10.0
AMOUNT_MIN, AMOUNT_MAX = 0.0, 20000.0

feature_names = X.columns.tolist()

with st.form("manual_predict_form"):
    col1, col2 = st.columns(2)
    time_val = col1.slider("⏱ Time (seconds)", TIME_MIN, TIME_MAX, int(df["Time"].mean()))
    amount_val = col2.slider("💰 Amount", AMOUNT_MIN, AMOUNT_MAX, float(df["Amount"].mean()))

    st.markdown("### PCA features V1–V28")
    cols_left, cols_right = st.columns(2)
    v_inputs = {}
    for i, fname in enumerate([f"V{i}" for i in range(1, 29)], start=1):
        default = float(df[fname].mean()) if fname in df.columns else 0.0
        if i <= 14:
            v_inputs[fname] = cols_left.slider(fname, float(V_MIN), float(V_MAX), float(default), key=f"v_{fname}")
        else:
            v_inputs[fname] = cols_right.slider(fname, float(V_MIN), float(V_MAX), float(default), key=f"v_{fname}")

    submit = st.form_submit_button("🔍 Predict Transaction")
    if submit:
        manual_dict = {"Time": float(time_val)}
        for i in range(1, 29):
            manual_dict[f"V{i}"] = float(v_inputs[f"V{i}"])
        manual_dict["Amount"] = float(amount_val)

        manual_df = pd.DataFrame([manual_dict])[feature_names]
        scaled = scaler.transform(manual_df)
        pred = model.predict(scaled)[0]
        proba = model.predict_proba(scaled)[0][1]

        st.markdown("### 🧾 Prediction Result")
        st.write(f"**Probability (fraud):** `{proba:.4f}`")
        if pred == 1:
            st.error("⚠️ Likely FRAUDULENT Transaction")
        else:
            st.success("✅ Likely Genuine Transaction")

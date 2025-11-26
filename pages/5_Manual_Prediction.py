# pages/5_Manual_Prediction.py
import streamlit as st
import pandas as pd
from utils import load_sample_csv, train_model_pipeline, ensure_feature_order


st.set_page_config(page_title="Manual Prediction", layout="wide")
st.markdown("# 🧮 Manual Transaction Prediction (Synthetic Inputs)")


try:
df = load_sample_csv()
except FileNotFoundError:
st.error("sample_creditcard.csv not found.")
st.stop()


metrics = train_model_pipeline(df)
feature_names = metrics['feature_names']


# slider ranges
TIME_MIN, TIME_MAX = 0, 172800
V_MIN, V_MAX = -10.0, 10.0
AMOUNT_MIN, AMOUNT_MAX = 0.0, 20000.0


with st.form('manual_predict'):
time_val = st.slider('Time (seconds)', TIME_MIN, TIME_MAX, int(df['Time'].mean()))
amount_val = st.slider('Amount', AMOUNT_MIN, AMOUNT_MAX, float(df['Amount'].mean()))


cols = st.columns(2)
inputs = {}
for i in range(1,29):
key = f'V{i}'
default = float(df[key].mean()) if key in df.columns else 0.0
col = cols[0] if i <= 14 else cols[1]
inputs[key] = col.slider(key, float(V_MIN), float(V_MAX), float(default))


submit = st.form_submit_button('Predict')
if submit:
manual_dict = {'Time': float(time_val)}
for i in range(1,29):
manual_dict[f'V{i}'] = float(inputs[f'V{i}'])
manual_dict['Amount'] = float(amount_val)


manual_df = pd.DataFrame([manual_dict])[feature_names]
scaler = metrics['scaler']
model = metrics['model']
scaled = scaler.transform(manual_df)
pred = model.predict(scaled)[0]
proba = model.predict_proba(scaled)[0][1]


st.write(f'**Probability (fraud):** {proba:.4f}')
if pred == 1:
st.error('⚠️ Likely FRAUDULENT Transaction')
else:
st.success('✅ Likely Genuine Transaction')
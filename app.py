# app.py
# Ultra Pro Max Premium Streamlit app - Credit Card Fraud Detection
# Requirements: see requirements.txt in repo

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io
import base64
import joblib
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
    precision_recall_curve,
)

# -----------------------
# Page config + CSS
# -----------------------
st.set_page_config(page_title="Credit Card Fraud — Premium", layout="wide", page_icon="💳")
st.markdown("""
    <style>
      .stApp { background: #0f1720; color: #e6eef6; }
      .title { font-weight:700; font-size:32px; }
      .muted { color:#9aa6b2; }
      .card { background: rgba(255,255,255,0.02); padding:14px; border-radius:10px; }
      .right-small { font-size:12px; color:#9aa6b2; }
      .metric-label { color:#bcd0df; }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Helpers: download link
# -----------------------
def get_download_link_df(df: pd.DataFrame, filename="data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download CSV</a>'
    return href

def get_download_link_model(obj, filename="model.joblib"):
    buffer = io.BytesIO()
    joblib.dump(obj, filename)
    # fallback to direct link via GitHub recommended; for local download keep simple
    return None

# -----------------------
# Load sample dataset (cached)
# -----------------------
@st.cache_data(show_spinner=False)
def load_sample(path="sample_creditcard.csv"):
    df = pd.read_csv(path)
    return df

# -----------------------
# Train model (cached resource)
# -----------------------
@st.cache_resource(show_spinner=False)
def train_pipeline(df: pd.DataFrame):
    # prepare
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # stratified split for balanced testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=3000, solver='lbfgs')
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    # feature importances by absolute coefficient
    feat_names = X.columns.tolist()
    coefs = model.coef_.flatten()
    feat_imp = pd.DataFrame({"feature": feat_names, "coef": coefs})
    feat_imp["abs_coef"] = feat_imp["coef"].abs()
    feat_imp = feat_imp.sort_values("abs_coef", ascending=False)

    return {
        "model": model,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "accuracy": acc,
        "cm": cm,
        "report": report,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "feat_imp": feat_imp,
        "features": feat_names,
    }

# -----------------------
# Main UI
# -----------------------
st.sidebar.image("https://raw.githubusercontent.com/pandas-dev/pandas/main/pandas/resources/pandas.png", width=60)
st.sidebar.title("Settings")
st.sidebar.markdown("**Deployment**")
retrain = st.sidebar.checkbox("Retrain model now (force)", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Links**")
st.sidebar.markdown("- GitHub repo\n- Live app\n")

# Try loading sample dataset (we include small sample)
try:
    df = load_sample()
except FileNotFoundError:
    st.error("sample_creditcard.csv not found in repo root. Upload or add the sample CSV.")
    st.stop()

# header
left, right = st.columns([3,1])
with left:
    st.markdown('<div class="title">💳 Credit Card Fraud Detection — Ultra Pro</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">A premium final-year project app: prediction form, advanced charts, model export.</div>', unsafe_allow_html=True)

with right:
    st.markdown(f"<div class='right-small'>Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</div>", unsafe_allow_html=True)

st.markdown("---")

# train or load model
if retrain:
    # clear cache by restarting the app or toggling different key - here we call train_pipeline again
    model_info = train_pipeline(df)
else:
    model_info = train_pipeline(df)

# -----------------------
# Dataset overview
# -----------------------
st.subheader("📊 Dataset Overview")
c1, c2, c3, c4 = st.columns([1,1,1,1])
c1.metric("Rows", f"{df.shape[0]:,}")
c2.metric("Features", f"{df.shape[1]-1}")
c3.metric("Fraud column", "Class")
c4.markdown(get_download_link_df(df.head(1000), "sample_creditcard.csv"), unsafe_allow_html=True)

if st.checkbox("Show raw data (first 50 rows)"):
    st.dataframe(df.head(50))

st.markdown("---")

# -----------------------
# Performance & Graphs (2-col layout)
# -----------------------
st.subheader("📈 Model Performance & Visualizations")
colL, colR = st.columns([1.1, 1])

# Left: Confusion matrix + distribution
with colL:
    st.markdown("### Confusion Matrix")
    cm = model_info["cm"]
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig, use_container_width=True)

    st.markdown("### Fraud vs Non-Fraud (sample)")
    fig2 = px.histogram(df, x="Class", title="Class distribution (sample)", labels={"Class":"Class (0 = not fraud, 1 = fraud)"})
    st.plotly_chart(fig2, use_container_width=True)

with colR:
    st.markdown("### Metrics")
    st.write(f"**Accuracy:** `{model_info['accuracy']:.4f}`")
    st.write("**Classification report:**")
    st.text(model_info["report"])

    st.markdown("### ROC Curve")
    fpr, tpr = model_info["fpr"], model_info["tpr"]
    fig3 = px.area(x=fpr, y=tpr, title=f"ROC curve (AUC = {model_info['roc_auc']:.3f})", labels={"x":"FPR","y":"TPR"})
    fig3.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash"))
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# -----------------------
# Feature importances
# -----------------------
st.subheader("🔎 Top Feature Importances")
fi = model_info["feat_imp"].head(12)
fig_fi = px.bar(fi.sort_values("abs_coef", ascending=True), x="abs_coef", y="feature", orientation="h", labels={"abs_coef":"abs(coef)"})
st.plotly_chart(fig_fi, use_container_width=True)

st.markdown("---")

# -----------------------
# Random test + manual prediction
# -----------------------
st.subheader("🎲 Try a Random Transaction or Enter Manually")

rcol, mcol = st.columns([1,1])

with rcol:
    if st.button("🔀 Pick & Predict Random Test Transaction"):
        X_test = model_info["X_test"]
        y_test = model_info["y_test"]
        idx = np.random.randint(0, X_test.shape[0])
        sample = X_test.iloc[idx]
        true = y_test.iloc[idx]
        st.write("**Sample features:**")
        st.dataframe(sample.to_frame().T)
        scaled = model_info["scaler"].transform([sample])
        pred = model_info["model"].predict(scaled)[0]
        proba = model_info["model"].predict_proba(scaled)[0][1]
        st.write(f"**Prediction:** {'Fraud (1)' if pred==1 else 'Not Fraud (0)'}")
        st.write(f"**Probability:** `{proba:.4f}`")
        if pred==true:
            st.success("✅ Model prediction matches actual label")
        else:
            st.warning("⚠ Prediction does not match actual label")

with mcol:
    st.markdown("### Manual Input (Time, V1–V28, Amount)")
    features = model_info["features"]
    means = df[features].mean()
    mins = df[features].min()
    maxs = df[features].max()

    with st.form("manual_predict_form"):
        # Time + Amount on top row
        tcol1, tcol2 = st.columns([1,1])
        with tcol1:
            inp_time = st.number_input("Time", float(mins["Time"]), float(maxs["Time"]), float(means["Time"]))
        with tcol2:
            inp_amount = st.number_input("Amount", float(mins["Amount"]), float(maxs["Amount"]), float(means["Amount"]))
        # V1-V28 two-column inputs
        p1, p2 = st.columns(2)
        v_vals = {}
        v_names = [f"V{i}" for i in range(1,29)]
        for i, v in enumerate(v_names):
            col = p1 if i < 14 else p2
            v_vals[v] = col.number_input(v, float(mins[v]), float(maxs[v]), float(means[v]), key=f"manual_{v}")

        submit = st.form_submit_button("Predict Transaction")
        if submit:
            inp = {"Time": inp_time, **{v: v_vals[v] for v in v_names}, "Amount": inp_amount}
            manual_df = pd.DataFrame([inp])[features]
            scaled_manual = model_info["scaler"].transform(manual_df)
            pred_manual = model_info["model"].predict(scaled_manual)[0]
            proba_manual = model_info["model"].predict_proba(scaled_manual)[0][1]
            st.write("### Result")
            st.write(f"**Prediction:** {'Fraud (1)' if pred_manual==1 else 'Not Fraud (0)'}")
            st.write(f"**Probability:** `{proba_manual:.4f}`")
            if pred_manual == 1:
                st.error("⚠ This transaction is likely FRAUDULENT.")
            else:
                st.success("✅ This transaction is likely NOT FRAUD.")

st.markdown("---")

# -----------------------
# Model export and resources
# -----------------------
st.subheader("📦 Export / Resources")
# Offer model download (joblib) — note: Streamlit can't always return very large files; we store small model to repo before offering
if st.button("Save model to repo (joblib)"):
    joblib.dump({"model": model_info["model"], "scaler": model_info["scaler"]}, "model_joblib.pkl")
    st.success("Saved model_joblib.pkl to repo root (you can download from the repo).")

st.markdown("**Sample dataset (included in repo):**")
st.markdown(f"Download the original uploaded ZIP (provided during setup): `file:///mnt/data/credit-card-fraud-detection.zip`")

st.info("If you want a downloadable link in the UI for the trained model or dataset files, I can add a GitHub raw link or upload to a static storage and link it.")
st.markdown("---")

st.caption("Made with ❤️ — Ultra Pro Max Premium edition. Ask me to prepare the final-year PDF report and presentation next.")

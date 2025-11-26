# pages/2_Model_Performance.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from utils import load_sample_csv, ensure_feature_order
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

st.set_page_config(layout="wide")
st.title("📈 Model Performance & Visualizations")

# Load data
try:
    df = load_sample_csv()
except FileNotFoundError:
    st.error("sample_creditcard.csv missing.")
    st.stop()

df = ensure_feature_order(df)

# Prepare data
X = df.drop("Class", axis=1)
y = df["Class"]

# split + scale + train (quick demo)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=2000, solver="lbfgs")
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:, 1]

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4)

# Show metrics
colL, colR = st.columns([1.2, 1])
with colL:
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white", fontsize=14)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig, use_container_width=True)

    st.subheader("Class distribution (sample)")
    fig_hist = px.histogram(df, x="Class", labels={"Class": "Class"}, title="Class distribution")
    st.plotly_chart(fig_hist, use_container_width=True)

with colR:
    st.subheader("Metrics")
    st.write(f"**Accuracy:** `{acc:.4f}`")
    st.text(report)

    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    roc_fig = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC = {roc_auc:.4f})", labels={"x": "FPR", "y": "TPR"})
    roc_fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash"))
    st.plotly_chart(roc_fig, use_container_width=True)

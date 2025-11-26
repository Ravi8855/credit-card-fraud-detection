# pages/4_Feature_Importances.py
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_sample_csv, ensure_feature_order
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(layout="wide")
st.title("🔎 Top Feature Importances (by |coef|)")

try:
    df = load_sample_csv()
except FileNotFoundError:
    st.error("sample_creditcard.csv missing.")
    st.stop()

df = ensure_feature_order(df)

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)

model = LogisticRegression(max_iter=2000, solver="lbfgs")
model.fit(X_train_s, y_train)

coefs = model.coef_.flatten()
feat_imp = pd.DataFrame({"feature": X.columns, "coef": coefs})
feat_imp["abs_coef"] = feat_imp["coef"].abs()
feat_imp = feat_imp.sort_values("abs_coef", ascending=False)

st.subheader("Top 12 features")
top12 = feat_imp.head(12).sort_values("abs_coef", ascending=True)
fig = px.bar(top12, x="abs_coef", y="feature", orientation="h", labels={"abs_coef": "abs(coef)"}, title="Top features by abs(coef)")
st.plotly_chart(fig, use_container_width=True)

st.write("Full coefficients table")
st.dataframe(feat_imp.reset_index(drop=True), use_container_width=True)

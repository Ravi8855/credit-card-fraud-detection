# =========================================================
# 🧮 OPTION 1 — Synthetic Manual Prediction Form (Premium)
# =========================================================

st.subheader("🧮 Manual Transaction Prediction (Synthetic Inputs)")

st.write("""
Enter synthetic (but realistic) values for a credit card transaction.
The model will calculate the fraud probability and prediction.
""")

# --- Reasonable ranges for synthetic values ---
TIME_MIN, TIME_MAX = 0, 172800
V_MIN, V_MAX = -10.0, 10.0
AMOUNT_MIN, AMOUNT_MAX = 0.0, 20000.0

st.markdown("### Enter Transaction Details")

with st.form("manual_form_option1"):
    # --- Top Row (Time & Amount) ---
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        time_val = st.slider("⏱ Time (0–172800 seconds)", TIME_MIN, TIME_MAX, 50000)

    with row1_col2:
        amount_val = st.slider("💰 Amount (₹0 – ₹20000)", AMOUNT_MIN, AMOUNT_MAX, 1200.0)

    st.markdown("### PCA Features (V1 – V28)")
    st.caption("These are PCA-transformed features (bank internal features). Synthetic values work perfectly.")

    # --- Display V1–V28 in two columns ---
    colA, colB = st.columns(2)
    v_values = {}

    for i in range(1, 29):
        col = colA if i <= 14 else colB
        v_values[f"V{i}"] = col.slider(
            f"V{i}",
            V_MIN,
            V_MAX,
            float(df[f"V{i}"].mean())  # default = dataset mean
        )

    submitted = st.form_submit_button("🔍 Predict Transaction")

    if submitted:
        input_dict = {
            "Time": float(time_val),
            **{f"V{i}": float(v_values[f"V{i}"]) for i in range(1, 29)},
            "Amount": float(amount_val),
        }

        manual_df = pd.DataFrame([input_dict])

        # Scale using fitted scaler
        scaled_manual = model_info["scaler"].transform(manual_df)

        # Predict
        pred_manual = model_info["model"].predict(scaled_manual)[0]
        proba_manual = model_info["model"].predict_proba(scaled_manual)[0][1]

        st.markdown("### 🧾 Prediction Result")

        # Output formatting
        if pred_manual == 1:
            st.error(f"⚠️ **Likely FRAUDULENT**  
            Probability: `{proba_manual:.4f}`")
        else:
            st.success(f"✅ **Likely Genuine Transaction**  
            Probability: `{proba_manual:.4f}`")

        st.info("Synthetic values do not represent real card data — used only for academic ML demo.")

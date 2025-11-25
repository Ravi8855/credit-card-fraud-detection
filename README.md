# 💳 Credit Card Fraud Detection (Machine Learning + Streamlit)

This project detects fraudulent credit card transactions using a Logistic Regression model trained on a **sample of the Kaggle Credit Card Fraud Dataset**.  
It includes:

- ✔ Clean ML pipeline  
- ✔ Logistic Regression model  
- ✔ Confusion matrix + classification report  
- ✔ Streamlit Web App (deployable online)  
- ✔ Final-year project friendly structure  

---

## 🧠 Project Overview

Credit card fraud is a major risk in financial systems.  
This project builds a **fraud detection model** that classifies transactions as:

- `0` → Genuine  
- `1` → Fraudulent  

The original full dataset is **284,807 rows**, but Streamlit Cloud cannot load 150MB files.  
So this project includes a **1000-row balanced sample**:

- 200 fraud cases  
- 800 normal cases  

Perfect for **fast online performance**.

---

## 📊 Dataset Details

- **Source:** Kaggle — Credit Card Fraud Detection  
- **Original Rows:** 284,807  
- **Sample Used:** 1000 rows (for Streamlit deployment)  
- **Features:**  
  - `Time`, `V1` to `V28` (PCA transformed)  
  - `Amount`  
  - `Class` (target)

⚠️ The **full dataset (`creditcard.csv`) is NOT included** in this repo due to GitHub’s 100MB limit.

However, this repo includes:


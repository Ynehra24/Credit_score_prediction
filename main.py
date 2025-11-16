import streamlit as st
import joblib
import numpy as np
import os
import gdown

def download_file_from_gdrive(file_id, filename):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    if not os.path.exists(filename):
        gdown.download(url, filename, quiet=False)

# Download both files BEFORE loading them
download_file_from_gdrive("1osTZxPqU5H204yKq1EHvCvw1DWM2Ejqy", "catboost_model_best.pkl")

# Now load the files
model = joblib.load("catboost_model_best.pkl")
scaler = joblib.load("scaler.pkl")

# Feature names
features = ['AMT_INCOME_TOTAL', 'X_Ratio', 'AGE']

# Weights: model=0.6, sentiment=0.15, employer=0.15, UPI=0.1 (sum to 1.0)
def predict_credit_score(X_input, sentiment_score, employer_review, upi_activity,
                        model_weight=0.6, sentiment_weight=0.15, employer_weight=0.15, upi_weight=0.1):
    X_scaled = scaler.transform(X_input)
    model_pred = model.predict(X_scaled)[0]
    final_score = (
        model_weight * model_pred +
        sentiment_weight * sentiment_score +
        employer_weight * employer_review +
        upi_weight * upi_activity
    )
    return final_score, model_pred

st.title("Credit Score Prediction Web App")

st.write("Enter the following details:")

user_inputs = []
for feat in features:
    val = st.number_input(f"{feat}:", min_value=0.0, value=0.0, step=0.01)
    user_inputs.append(val)
sentiment_score = st.slider("Social Media Sentiment Score (0-100):", 0.0, 100.0, 50.0, 0.1)
employer_review = st.slider("Employer Review Score (0-100):", 0.0, 100.0, 50.0, 0.1)
upi_activity = st.slider("UPI Activity Score (0-100):", 0.0, 100.0, 50.0, 0.1)

if st.button("Predict Credit Score"):
    X_input = np.array(user_inputs).reshape(1, -1)
    final_score, model_pred = predict_credit_score(X_input, sentiment_score, employer_review, upi_activity)
    st.success(f"Model Credit Score: {model_pred:.2f}")
    st.info(f"Social Media Score: {sentiment_score:.2f}")
    st.info(f"Employer Review Score: {employer_review:.2f}")
    st.info(f"UPI Activity Score: {upi_activity:.2f}")
    st.header(f"Final Weighted Credit Score: {final_score:.2f}")

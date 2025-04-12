# app.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load model and feature names
model = joblib.load("models/model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# Set up page
st.set_page_config(page_title="Telco Churn Predictor", layout="centered")
st.title("📊 Telco Customer Churn Predictor")

# Label encoding function
def encode_inputs(df):
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

# Input Form
with st.form("churn_form"):
    st.subheader("Enter Customer Details")

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 3000.0)

    submitted = st.form_submit_button("Predict Churn")

# Process input
if submitted:
    input_df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    # Encode and align
    encoded = encode_inputs(input_df)
    for col in feature_names:
        if col not in encoded.columns:
            encoded[col] = 0
    encoded = encoded[feature_names]  # align order

    # Predict
    prediction = model.predict(encoded)[0]
    prob = model.predict_proba(encoded)[0][1]
    result = "🟢 Customer is **not likely to churn**" if prediction == 0 else "🔴 Customer is **likely to churn**"

    # Display
    st.markdown(f"### 🔍 Prediction Result: {result}")
    st.markdown(f"#### 💡 Probability of churn: `{prob:.2%}`")

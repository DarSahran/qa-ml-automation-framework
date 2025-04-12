import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os

# Page setup
st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")

# Load model + features safely
try:
    model = joblib.load("models/model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
except Exception as e:
    st.error("⚠️ Model files not found. Please train the model first.")
    st.stop()

# Label encoding
def encode_inputs(df):
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

# ---------------- SIDEBAR ----------------
st.sidebar.header("🧾 Customer Profile")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 3000.0)

# ---------------- MAIN ----------------
st.title("📊 Telco Customer Churn Prediction")
st.markdown("This dashboard estimates churn probability based on customer profile inputs.")

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

# Encode & align features
encoded = encode_inputs(input_df)
for col in feature_names:
    if col not in encoded.columns:
        encoded[col] = 0
encoded = encoded[feature_names]

# Predict
prediction = model.predict(encoded)[0]
prob = model.predict_proba(encoded)[0][1]

# Display
st.subheader("🔍 Prediction Result")
st.markdown(
    f"**{'🟢 Customer is not likely to churn' if prediction == 0 else '🔴 Customer is likely to churn'}**"
)
st.metric(label="Churn Probability", value=f"{prob:.2%}")
st.progress(int(prob * 100))

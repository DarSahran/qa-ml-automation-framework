import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load model and features
model = joblib.load("models/model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# Encode categorical inputs
def encode_inputs(df):
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

# Page setup
st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
    }
    .metric-box {
        background-color: #f4f6fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 6px rgba(0,0,0,0.1);
    }
    .section-title {
        font-size: 22px;
        font-weight: 600;
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- Sidebar -------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Telecom_Icon.svg/1024px-Telecom_Icon.svg.png", width=120)
st.sidebar.header("📋 Input Customer Info")

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

# ----------------- Main Dashboard -------------------
st.title("📊 Telco Customer Churn Dashboard")
st.markdown("---")

col1, col2 = st.columns(2)

# Predict only when page loads or inputs change
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

encoded = encode_inputs(input_df)
for col in feature_names:
    if col not in encoded.columns:
        encoded[col] = 0
encoded = encoded[feature_names]

prediction = model.predict(encoded)[0]
prob = model.predict_proba(encoded)[0][1]

# Results in layout boxes
with col1:
    st.markdown("<div class='section-title'>Prediction Result</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='metric-box'>
        {'🟢 <b>Customer is NOT likely to churn</b>' if prediction == 0 else '🔴 <b>Customer is LIKELY to churn</b>'}<br><br>
        <span style='font-size: 16px;'>Probability of churn: <b>{prob:.2%}</b></span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("<div class='section-title'>Probability Meter</div>", unsafe_allow_html=True)
    st.progress(int(prob * 100))

# scripts/predict.py

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

def load_model(model_path="models/model.pkl"):
    return joblib.load(model_path)

def predict(model, data_path="data/Telco_Customer_Churn.csv"):
    df = pd.read_csv(data_path)
    df.drop(columns=["customerID", "PaymentMethod"], inplace=True, errors="ignore")
    df.dropna(inplace=True)

    # Encode target
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    # Label encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop("Churn", axis=1)
    return model.predict(X)

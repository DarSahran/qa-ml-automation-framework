# scripts/train.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

def train_model(data_path="data/Telco_Customer_Churn.csv", model_path="models/model.pkl", random_state=42):
    df = pd.read_csv(data_path)

    # Drop customerID and PaymentMethod
    df.drop(columns=["customerID", "PaymentMethod"], inplace=True, errors="ignore")

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Encode target
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    # Label encode all categorical features
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    # Features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Train model
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save model
    joblib.dump(model, model_path)

    return accuracy

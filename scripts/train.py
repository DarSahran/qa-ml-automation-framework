# scripts/train.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_model(data_path="data/sample_data.csv", model_path="models/model.pkl", random_state=42):
    df = pd.read_csv(data_path)
    
    X = df[["age"]]  # Minimal feature for now
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)

    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(model, model_path)
    return accuracy

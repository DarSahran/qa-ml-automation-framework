# scripts/predict.py

import pandas as pd
import joblib

def load_model(model_path="models/model.pkl"):
    return joblib.load(model_path)

def predict(model, data_path="data/sample_data.csv"):
    df = pd.read_csv(data_path)
    X = df[["age"]]  # Same feature used in training
    return model.predict(X)

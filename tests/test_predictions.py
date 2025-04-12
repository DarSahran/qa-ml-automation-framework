# tests/test_predictions.py

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.predict import load_model, predict

DATA_PATH = "data/Telco_Customer_Churn.csv"

def test_model_loads():
    model = load_model()
    assert model is not None, "Model failed to load."

def test_prediction_output_shape():
    df = pd.read_csv(DATA_PATH)
    df.drop(columns=["customerID", "PaymentMethod"], inplace=True, errors="ignore")
    df.dropna(inplace=True)

    expected_rows = df.shape[0]
    model = load_model()
    preds = predict(model)
    
    assert preds.shape[0] == expected_rows, f"Expected {expected_rows} predictions, got {preds.shape[0]}"

def test_prediction_classes():
    model = load_model()
    preds = predict(model)
    assert set(np.unique(preds)).issubset({0, 1}), f"Predictions contain unexpected values: {set(preds)}"

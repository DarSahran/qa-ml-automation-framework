# tests/test_predictions.py

import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.predict import load_model, predict

def test_model_loads():
    model = load_model()
    assert model is not None, "Model failed to load."

def test_prediction_output_shape():
    model = load_model()
    preds = predict(model)
    assert preds.shape[0] == 5000, f"Expected 5000 predictions, got {preds.shape[0]}"

def test_prediction_classes():
    model = load_model()
    preds = predict(model)
    assert set(np.unique(preds)).issubset({0, 1}), f"Predictions contain unexpected values: {set(preds)}"

# tests/test_model_training.py

import sys
import os

# Add root path to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

# Now import works
from scripts.train import train_model
import os

def test_model_training_runs():
    accuracy = train_model()
    assert isinstance(accuracy, float), "Training did not return accuracy as float."

def test_accuracy_above_threshold():
    accuracy = train_model()
    assert accuracy >= 0.65, f"Model accuracy {accuracy:.2f} below expected threshold of 0.6"

def test_model_file_created():
    model_path = "models/model.pkl"
    train_model(model_path=model_path)
    assert os.path.exists(model_path), "Model file not found after training."

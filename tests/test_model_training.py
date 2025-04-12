import sys
import os

# Add root path to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from scripts.train import train_model

def test_model_training_runs():
    accuracy = train_model()
    assert isinstance(accuracy, float), "Training did not return accuracy as float."

def test_accuracy_above_threshold():
    accuracy = train_model()
    assert accuracy >= 0.65, f"Model accuracy {accuracy:.2f} below expected threshold of 0.65"

def test_model_file_created():
    model_path = "models/model.pkl"
    train_model(model_path=model_path)
    assert os.path.exists(model_path), "Model file not found after training."

def test_feature_names_file_created():
    feature_path = "models/feature_names.pkl"
    train_model()
    assert os.path.exists(feature_path), "Feature names file not found after training."

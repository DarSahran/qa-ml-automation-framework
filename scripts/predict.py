# scripts/predict.py

import pandas as pd
import joblib

def load_model(model_path="models/model.pkl"):
    return joblib.load(model_path)

def predict(model, data_path="data/sample_data.csv"):
    df = pd.read_csv(data_path)
    X = df[["age","country"]]  # Same feature used in trainingimport pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Function to load the trained model
def load_model(model_path="models/model.pkl"):
    return joblib.load(model_path)

# Function to encode the 'country' column (using Label Encoding)
def encode_country(df):
    # Initialize the label encoder
    label_encoder = LabelEncoder()
    
    # Encode the 'country' column
    df['country'] = label_encoder.fit_transform(df['country'])
    
    return df

def predict(model, data_path="data/sample_data.csv"):
    df = pd.read_csv(data_path)
    
    # Apply encoding on the 'country' column
    df = encode_country(df)
    
    # Prepare features with encoded 'country'
    X = df[["age", "country"]]  # Using the encoded 'country' column
    return model.predict(X)

    return model.predict(X)

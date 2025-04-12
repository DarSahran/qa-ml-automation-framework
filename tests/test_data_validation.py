# tests/test_data_validation.py

import pandas as pd
import yaml
import pytest

CONFIG_PATH = "configs/test_config.yaml"
DATA_PATH = "data/Telco_Customer_Churn.csv"  # You'll create this later for testing

@pytest.fixture(scope="module")
def config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA_PATH)

def test_required_columns_present(df, config):
    required = set(config['data_validation']['required_columns'])
    actual = set(df.columns)
    missing = required - actual
    assert not missing, f"Missing columns: {missing}"

def test_null_percentage(df, config):
    threshold = config['data_validation']['max_null_percentage']
    null_percent = df.isnull().mean() * 100
    over_limit = null_percent[null_percent > threshold]
    assert over_limit.empty, f"Too many nulls in: {list(over_limit.index)}"

def test_column_dtypes(df, config):
    expected_dtypes = config['data_validation']['expected_dtypes']
    for col, expected_dtype in expected_dtypes.items():
        if col in df.columns:
            actual_dtype = str(df[col].dtype)
            assert actual_dtype == expected_dtype, f"{col} has {actual_dtype}, expected {expected_dtype}"

import pickle
import pandas as pd
import os
import pytest
import numpy as np


DATA_RAW_DTYPES_OPTIMIZED = {
        "key": "O",
        "fare_amount": "float32",
        "pickup_datetime": "O",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int8"
    }


@pytest.fixture(scope="session")  # cached fixture
def train_1k()->pd.DataFrame:

    aws_path = "https://wagon-public-datasets.s3.amazonaws.com/taxi-fare-ny/train_1k.csv"
    df_raw = pd.read_csv(aws_path, dtype=DATA_RAW_DTYPES_OPTIMIZED)

    return df_raw


@pytest.fixture(scope='session')
def train_1k_cleaned()->pd.DataFrame:
    aws_path = "https://wagon-public-datasets.s3.amazonaws.com/taxi-fare-ny/solutions/train_1k_cleaned.csv"
    df_cleaned = pd.read_csv(aws_path, dtype=DATA_RAW_DTYPES_OPTIMIZED)

    return df_cleaned


@pytest.fixture(scope='session')
def X_processed_1k() -> np.ndarray:
    with open(os.path.join(os.path.dirname(__file__), "fixtures", "X_processed_1k.npy"), "rb") as f:
        X_processed_1k = np.load(f)
    return X_processed_1k


@pytest.fixture(scope='session')
def y_1k() -> pd.Series:
    with open(os.path.join(os.path.dirname(__file__), "fixtures", "y_1k.pickle"), "rb") as f:
        y = pickle.load(f)
    return y

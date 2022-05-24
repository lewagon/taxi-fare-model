"""
taxifare model package params
load and validate the environment variables in the `.env`
"""

import os

import numpy as np

# °º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸

DATASET_SIZE = os.environ.get("DATASET_SIZE")
VALIDATION_DATASET_SIZE = os.environ.get("VALIDATION_DATASET_SIZE")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))

PROJECT = os.environ.get("PROJECT")
DATASET = os.environ.get("DATASET")

# °º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸

env_valid_options = dict(
    DATASET_SIZE=["1k", "10k", "100k", "500k", "50M"],
    VALIDATION_DATASET_SIZE=["1k", "10k", "100k", "500k", "500k"],
    CHUNK_SIZE=["200", "2000", "20000", "100000", "1000000"],
    DATA_SOURCE=["local", "big query"],
    MODEL_TARGET=["local", "gcs", "mlflow"],
    PREFECT_BACKEND=["local", "server", "cloud"])


def validate_env_value(env, valid_options):
    env_value = os.environ[env]
    if env_value not in valid_options:
        raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")


for env, valid_options in env_valid_options.items():
    validate_env_value(env, valid_options)

# °º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸


ROOT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

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

DATA_PROCESSED_DTYPES_OPTIMIZED = np.float32

DATA_RAW_COLUMNS = DATA_RAW_DTYPES_OPTIMIZED.keys()

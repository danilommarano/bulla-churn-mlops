"""KFP step: evaluate a persisted model on the test parquet and write metrics JSON."""

import json

import joblib
import pandas as pd

from churn.features.builder import INPUT_COLUMNS
from churn.training.evaluate import evaluate


def evaluate_model(model_path: str, test_path: str, metrics_out: str) -> dict:
    """Load the model, evaluate on the test parquet, write metrics as JSON, and return them."""
    pipeline = joblib.load(model_path)
    df = pd.read_parquet(test_path)
    X, y = df[INPUT_COLUMNS], df["turnover"]
    metrics = evaluate(pipeline, X, y)
    with open(metrics_out, "w") as f:
        json.dump(metrics, f)
    return metrics

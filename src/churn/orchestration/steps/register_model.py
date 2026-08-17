"""KFP step: register the model in MLflow, promoting to @production only if the gate passes."""

import json

import joblib
import pandas as pd

from churn.config import Settings, settings
from churn.features.builder import INPUT_COLUMNS
from churn.training.registry import log_and_register


def register_model(
    model_path: str, metrics_path: str, train_path: str, cfg: Settings = settings
) -> dict:
    """Log + register the model; move the @production alias only if roc_auc >= cfg.min_roc_auc.

    Returns run_id, version, promoted, roc_auc.
    """
    pipeline = joblib.load(model_path)  # trusted: artifact written by train_model step in this same pipeline
    with open(metrics_path) as f:
        metrics = json.load(f)
    X_sample = pd.read_parquet(train_path)[INPUT_COLUMNS].head(5)

    roc_auc = metrics["roc_auc"]
    promote = roc_auc >= cfg.min_roc_auc
    result = log_and_register(pipeline, X_sample, metrics, cfg, promote=promote)
    return {**result, "roc_auc": roc_auc}

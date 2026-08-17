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


class ModelGateError(RuntimeError):
    """Raised when the trained model fails the quality gate (roc_auc below min_roc_auc)."""


def require_promotion(result: dict) -> None:
    """Fail the pipeline if the model was not promoted (quality gate rejected it).

    The pure `register_model` step reports promotion as data; the orchestration layer
    turns a rejection into a hard failure so CI (and Vertex-style model validation)
    fails the run instead of silently shipping a bad model.
    """
    if not result["promoted"]:
        raise ModelGateError(
            f"Quality gate rejected the model: roc_auc={result['roc_auc']:.4f} "
            "below min_roc_auc; not promoted to @production."
        )

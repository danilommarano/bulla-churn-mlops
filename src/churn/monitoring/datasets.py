# src/churn/monitoring/datasets.py
"""Build the reference and current datasets for a monitoring run.

Reference = train split (baseline the model was fit on). Current = holdout
(X_test), optionally perturbed with simulate_drift to demonstrate detection.
Each frame carries the 12 raw INPUT_COLUMNS + prob_churn (model score) + turnover.
The split reproduces training exactly (same random_state / test_size / stratify).
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import Settings, settings
from churn.data import load_raw
from churn.features.builder import INPUT_COLUMNS
from churn.monitoring.perturb import simulate_drift


def _scored(X: pd.DataFrame, y: pd.Series, model) -> pd.DataFrame:
    frame = X[INPUT_COLUMNS].copy()
    frame["prob_churn"] = model.predict_proba(frame[INPUT_COLUMNS])[:, 1]
    frame["turnover"] = y.to_numpy()
    return frame


def build_reference_current(
    cfg: Settings = settings, *, model, simulate: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (reference, current) scored frames for a monitoring run."""
    df = load_raw(cfg.data_path)
    X, y = df[INPUT_COLUMNS], df["turnover"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    if simulate:
        X_test = simulate_drift(X_test)
    reference = _scored(X_train, y_train, model)
    current = _scored(X_test, y_test, model)
    return reference, current

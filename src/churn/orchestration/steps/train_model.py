"""KFP step: fit the churn pipeline on the train parquet and persist it with joblib."""

import joblib
import pandas as pd

from churn.features.builder import INPUT_COLUMNS
from churn.training.pipeline import build_pipeline


def train_model(
    train_path: str, model_out: str, random_state: int, n_age_bins: int
) -> None:
    """Fit build_pipeline() on the train parquet and joblib.dump the fitted Pipeline to model_out.

    joblib is used for ML model serialization (scikit-learn Pipeline with fitted transformers).
    """
    df = pd.read_parquet(train_path)
    X, y = df[INPUT_COLUMNS], df["turnover"]
    pipeline = build_pipeline(random_state=random_state, n_age_bins=n_age_bins)
    pipeline.fit(X, y)
    joblib.dump(pipeline, model_out)

"""Shared MLflow logging + registration for the churn model.

DRY helper used by both the training entrypoint (`train.py`) and the KFP
`register_model` step, so the two never drift in how they log/register.
"""

import warnings

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

from churn.config import Settings, settings
from churn.features.builder import INPUT_COLUMNS


def log_and_register(
    pipeline, X_sample, metrics: dict, cfg: Settings = settings, promote: bool = True
) -> dict:
    """Log the fitted pipeline + params/metrics to MLflow, register a model version,
    and (optionally) move the production alias.

    `X_sample` is a DataFrame with INPUT_COLUMNS used for the signature and input example
    (a few rows suffice — the signature is schema-based). Returns run_id, version, promoted.
    """
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "model": "LogisticRegression",
                "max_iter": 500,
                "class_weight": "balanced",
                "random_state": cfg.random_state,
                "test_size": cfg.test_size,
                "n_age_bins": cfg.n_age_bins,
                "n_input_features": len(INPUT_COLUMNS),
            }
        )
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "confusion_matrix"})
        signature = infer_signature(X_sample, pipeline.predict(X_sample))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # mute pip-version + int-column hints
            info = mlflow.sklearn.log_model(
                pipeline,
                name="model",
                signature=signature,
                input_example=X_sample.head(3),
                registered_model_name=cfg.model_name,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
        version = str(info.registered_model_version)
        if promote:
            MlflowClient(tracking_uri=cfg.mlflow_tracking_uri).set_registered_model_alias(
                cfg.model_name, cfg.model_alias, version
            )
        run_id = run.info.run_id

    return {"run_id": run_id, "version": version, "promoted": promote}

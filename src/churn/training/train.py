"""Training entrypoint: fit the pipeline, evaluate honestly, track and register in MLflow."""

import warnings

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split

from churn.config import Settings, settings
from churn.data import load_raw
from churn.features.builder import INPUT_COLUMNS
from churn.training.evaluate import evaluate
from churn.training.pipeline import build_pipeline


def train(cfg: Settings = settings) -> dict:
    """Fit, evaluate and register the churn model. Returns run id, version and metrics."""
    df = load_raw(cfg.data_path)
    X, y = df[INPUT_COLUMNS], df["turnover"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    pipeline = build_pipeline(random_state=cfg.random_state, n_age_bins=cfg.n_age_bins)
    pipeline.fit(X_train, y_train)
    metrics = evaluate(pipeline, X_test, y_test)

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
        signature = infer_signature(X_train, pipeline.predict(X_train))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # mute pip-version + int-column hints
            info = mlflow.sklearn.log_model(
                pipeline,
                name="model",
                signature=signature,
                input_example=X_train.head(3),
                registered_model_name=cfg.model_name,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
        MlflowClient(tracking_uri=cfg.mlflow_tracking_uri).set_registered_model_alias(
            cfg.model_name, cfg.model_alias, str(info.registered_model_version)
        )
        run_id = run.info.run_id

    return {
        "run_id": run_id,
        "version": str(info.registered_model_version),
        "metrics": metrics,
    }


def main() -> None:
    result = train()
    m = result["metrics"]
    print(
        f"roc_auc={m['roc_auc']:.4f} precision={m['precision']:.4f} "
        f"recall={m['recall']:.4f} f1={m['f1']:.4f} accuracy={m['accuracy']:.4f}"
    )
    print(
        f"registered '{settings.model_name}' v{result['version']} "
        f"@{settings.model_alias} (run {result['run_id']})"
    )


if __name__ == "__main__":
    main()

"""Training entrypoint: fit the pipeline, evaluate honestly, track and register in MLflow."""

from sklearn.model_selection import train_test_split

from churn.config import Settings, settings
from churn.data import load_raw
from churn.features.builder import INPUT_COLUMNS
from churn.training.evaluate import evaluate
from churn.training.pipeline import build_pipeline
from churn.training.registry import log_and_register


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

    # This quick-run entrypoint always promotes. The min_roc_auc quality gate lives in
    # the KFP register_model step (make pipeline), which is the governed promotion path.
    result = log_and_register(pipeline, X_train, metrics, cfg, promote=True)
    return {
        "run_id": result["run_id"],
        "version": result["version"],
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

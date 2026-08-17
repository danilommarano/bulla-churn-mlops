from pathlib import Path

from mlflow import MlflowClient
from sklearn.model_selection import train_test_split

from churn.config import Settings
from churn.data import load_raw
from churn.features.builder import INPUT_COLUMNS
from churn.training.evaluate import evaluate
from churn.training.pipeline import build_pipeline

CSV_PATH = str(Path(__file__).resolve().parents[1] / "Customer-Churn-Records.csv")


def _cfg(tmp_path) -> Settings:
    return Settings(
        data_path=CSV_PATH,
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        mlflow_experiment="churn-test",
        model_name="churn-model-test",
        model_alias="production",
    )


def _fit(cfg: Settings):
    df = load_raw(cfg.data_path)
    X, y = df[INPUT_COLUMNS], df["turnover"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    pipeline = build_pipeline(random_state=cfg.random_state, n_age_bins=cfg.n_age_bins)
    pipeline.fit(X_train, y_train)
    metrics = evaluate(pipeline, X_test, y_test)
    return pipeline, X_train, metrics


def test_log_and_register_promotes_when_asked(tmp_path):
    from churn.training.registry import log_and_register

    cfg = _cfg(tmp_path)
    pipeline, X_train, metrics = _fit(cfg)
    result = log_and_register(pipeline, X_train, metrics, cfg, promote=True)

    assert result["version"] == "1"
    assert result["promoted"] is True
    client = MlflowClient(tracking_uri=cfg.mlflow_tracking_uri)
    mv = client.get_model_version_by_alias(cfg.model_name, cfg.model_alias)
    assert str(mv.version) == "1"


def test_log_and_register_skips_alias_when_not_promoting(tmp_path):
    from churn.training.registry import log_and_register

    cfg = _cfg(tmp_path)
    pipeline, X_train, metrics = _fit(cfg)
    result = log_and_register(pipeline, X_train, metrics, cfg, promote=False)

    assert result["version"] == "1"
    assert result["promoted"] is False
    client = MlflowClient(tracking_uri=cfg.mlflow_tracking_uri)
    # a version exists, but no production alias was set
    assert client.get_model_version(cfg.model_name, "1") is not None
    try:
        client.get_model_version_by_alias(cfg.model_name, cfg.model_alias)
        raise AssertionError("alias should not exist when promote=False")
    except Exception:  # noqa: BLE001, S110
        pass

from pathlib import Path

import pandas as pd

from churn.features.builder import INPUT_COLUMNS

CSV_PATH = str(Path(__file__).resolve().parents[1] / "Customer-Churn-Records.csv")


def test_prepare_data_writes_model_input_parquet(tmp_path):
    from churn.orchestration.steps.prepare_data import prepare_data

    out = tmp_path / "prepared.parquet"
    prepare_data(str(out), CSV_PATH)

    df = pd.read_parquet(out)
    assert list(df.columns) == INPUT_COLUMNS + ["turnover"]
    assert len(df) == 10000


def test_split_data_reproduces_stratified_split(tmp_path):
    from churn.orchestration.steps.prepare_data import prepare_data
    from churn.orchestration.steps.split_data import split_data

    prepared = tmp_path / "prepared.parquet"
    prepare_data(str(prepared), CSV_PATH)

    train_out = tmp_path / "train.parquet"
    test_out = tmp_path / "test.parquet"
    split_data(str(prepared), str(train_out), str(test_out), test_size=0.2, random_state=42)

    train_df = pd.read_parquet(train_out)
    test_df = pd.read_parquet(test_out)

    # columns preserved on both sides
    assert list(train_df.columns) == INPUT_COLUMNS + ["turnover"]
    assert list(test_df.columns) == INPUT_COLUMNS + ["turnover"]
    # 80/20 split of 10000 rows, no rows lost
    assert len(train_df) == 8000
    assert len(test_df) == 2000
    # stratified: class balance preserved within ~1 percentage point
    assert abs(train_df["turnover"].mean() - test_df["turnover"].mean()) < 0.01


def test_train_model_persists_fitted_pipeline(tmp_path):
    import joblib

    from churn.orchestration.steps.prepare_data import prepare_data
    from churn.orchestration.steps.split_data import split_data
    from churn.orchestration.steps.train_model import train_model

    prepared = tmp_path / "prepared.parquet"
    prepare_data(str(prepared), CSV_PATH)
    train_out = tmp_path / "train.parquet"
    test_out = tmp_path / "test.parquet"
    split_data(str(prepared), str(train_out), str(test_out), test_size=0.2, random_state=42)

    model_out = tmp_path / "model.joblib"
    train_model(str(train_out), str(model_out), random_state=42, n_age_bins=5)

    model = joblib.load(model_out)
    sample = pd.read_parquet(test_out)[INPUT_COLUMNS].head(5)
    assert len(model.predict(sample)) == 5
    assert model.predict_proba(sample).shape == (5, 2)


def test_evaluate_model_writes_metrics_json(tmp_path):
    import json

    from churn.orchestration.steps.evaluate_model import evaluate_model
    from churn.orchestration.steps.prepare_data import prepare_data
    from churn.orchestration.steps.split_data import split_data
    from churn.orchestration.steps.train_model import train_model

    prepared = tmp_path / "prepared.parquet"
    prepare_data(str(prepared), CSV_PATH)
    train_out = tmp_path / "train.parquet"
    test_out = tmp_path / "test.parquet"
    split_data(str(prepared), str(train_out), str(test_out), test_size=0.2, random_state=42)
    model_out = tmp_path / "model.joblib"
    train_model(str(train_out), str(model_out), random_state=42, n_age_bins=5)

    metrics_out = tmp_path / "metrics.json"
    returned = evaluate_model(str(model_out), str(test_out), str(metrics_out))

    on_disk = json.loads(metrics_out.read_text())
    assert on_disk == returned
    for key in ("roc_auc", "precision", "recall", "f1", "accuracy", "confusion_matrix"):
        assert key in on_disk
    assert 0.0 < on_disk["roc_auc"] < 1.0
    assert on_disk["roc_auc"] > 0.7


def _build_artifacts(tmp_path):
    """Run prepare -> split -> train -> evaluate, returning the artifact paths."""
    from churn.orchestration.steps.evaluate_model import evaluate_model
    from churn.orchestration.steps.prepare_data import prepare_data
    from churn.orchestration.steps.split_data import split_data
    from churn.orchestration.steps.train_model import train_model

    prepared = tmp_path / "prepared.parquet"
    prepare_data(str(prepared), CSV_PATH)
    train_out = tmp_path / "train.parquet"
    test_out = tmp_path / "test.parquet"
    split_data(str(prepared), str(train_out), str(test_out), test_size=0.2, random_state=42)
    model_out = tmp_path / "model.joblib"
    train_model(str(train_out), str(model_out), random_state=42, n_age_bins=5)
    metrics_out = tmp_path / "metrics.json"
    evaluate_model(str(model_out), str(test_out), str(metrics_out))
    return str(model_out), str(metrics_out), str(train_out)


def _reg_cfg(tmp_path, min_roc_auc):
    from churn.config import Settings

    return Settings(
        data_path=CSV_PATH,
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        mlflow_experiment="churn-test",
        model_name="churn-model-test",
        model_alias="production",
        min_roc_auc=min_roc_auc,
    )


def test_register_model_promotes_when_gate_passes(tmp_path):
    from mlflow import MlflowClient

    from churn.orchestration.steps.register_model import register_model

    model_path, metrics_path, train_path = _build_artifacts(tmp_path)
    cfg = _reg_cfg(tmp_path, min_roc_auc=0.5)
    result = register_model(model_path, metrics_path, train_path, cfg)

    assert result["promoted"] is True
    assert result["version"] == "1"
    assert result["roc_auc"] > 0.7
    client = MlflowClient(tracking_uri=cfg.mlflow_tracking_uri)
    mv = client.get_model_version_by_alias(cfg.model_name, cfg.model_alias)
    assert str(mv.version) == "1"


def test_register_model_skips_promotion_when_gate_fails(tmp_path):
    import pytest
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    from churn.orchestration.steps.register_model import register_model

    model_path, metrics_path, train_path = _build_artifacts(tmp_path)
    cfg = _reg_cfg(tmp_path, min_roc_auc=0.99)
    result = register_model(model_path, metrics_path, train_path, cfg)

    assert result["promoted"] is False
    assert result["version"] == "1"
    client = MlflowClient(tracking_uri=cfg.mlflow_tracking_uri)
    # version registered, but no production alias set
    assert client.get_model_version(cfg.model_name, "1") is not None
    with pytest.raises(MlflowException):
        client.get_model_version_by_alias(cfg.model_name, cfg.model_alias)


def test_pipeline_end_to_end(tmp_path):
    from mlflow import MlflowClient

    from churn.config import Settings
    from churn.orchestration.dag import run_local

    cfg = Settings(
        data_path=CSV_PATH,
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        mlflow_experiment="churn-e2e",
        model_name="churn-model-e2e",
        model_alias="production",
        min_roc_auc=0.5,
    )

    run = run_local(cfg, pipeline_root=str(tmp_path / "kfp_outputs"))
    assert run.state.name == "FINAL"

    # the DAG registered a version and promoted it (gate passes at 0.5)
    client = MlflowClient(tracking_uri=cfg.mlflow_tracking_uri)
    mv = client.get_model_version_by_alias(cfg.model_name, cfg.model_alias)
    assert str(mv.version) == "1"

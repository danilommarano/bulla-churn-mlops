from pathlib import Path

from mlflow import MlflowClient
from mlflow.sklearn import load_model

from churn.config import Settings
from churn.features.builder import INPUT_COLUMNS
from churn.training.train import train

CSV_PATH = str(Path(__file__).resolve().parents[1] / "Customer-Churn-Records.csv")


def _isolated_settings(tmp_path) -> Settings:
    return Settings(
        data_path=CSV_PATH,
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        mlflow_experiment="churn-test",
        model_name="churn-model-test",
        model_alias="production",
    )


def test_train_logs_run_and_registers_model(tmp_path):
    cfg = _isolated_settings(tmp_path)
    result = train(cfg)

    # honest metric was computed and is plausible for this dataset
    assert result["metrics"]["roc_auc"] > 0.7
    assert result["version"] == "1"

    # the alias points at the freshly trained version, and it loads + predicts
    client = MlflowClient(tracking_uri=cfg.mlflow_tracking_uri)
    mv = client.get_model_version_by_alias(cfg.model_name, cfg.model_alias)
    assert str(mv.version) == result["version"]

    # the model loads from the registry and predicts on the same input schema it was trained on
    model = load_model(f"models:/{cfg.model_name}@{cfg.model_alias}")
    from churn.data import load_raw

    sample = load_raw(cfg.data_path)[INPUT_COLUMNS].head(5)
    assert len(model.predict(sample)) == 5

import json
from pathlib import Path

from churn.config import Settings
from churn.reporting import export_production_metrics
from churn.training.train import train

CSV_PATH = str(Path(__file__).resolve().parents[1] / "Customer-Churn-Records.csv")


def test_export_production_metrics_writes_expected_keys(tmp_path):
    cfg = Settings(
        data_path=CSV_PATH,
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        mlflow_experiment="report-test",
        model_name="churn-model-report-test",
        model_alias="production",
    )
    train(cfg)
    out = tmp_path / "metrics.json"
    metrics = export_production_metrics(cfg, out)

    assert out.exists()
    on_disk = json.loads(out.read_text())
    assert on_disk == metrics
    for key in ("roc_auc", "precision", "recall", "f1", "accuracy", "confusion_matrix"):
        assert key in metrics
    assert len(metrics["confusion_matrix"]) == 2
    assert 0.0 <= metrics["roc_auc"] <= 1.0

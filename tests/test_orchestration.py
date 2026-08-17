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

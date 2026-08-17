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

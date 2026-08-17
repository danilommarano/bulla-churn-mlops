"""KFP step: load + validate the raw CSV and persist the model-input frame as parquet."""

from churn.data import load_raw
from churn.features.builder import INPUT_COLUMNS


def prepare_data(out_path: str, data_path: str) -> None:
    """Write INPUT_COLUMNS + turnover from the validated raw frame to a parquet at out_path."""
    df = load_raw(data_path)
    df[INPUT_COLUMNS + ["turnover"]].to_parquet(out_path, index=False)

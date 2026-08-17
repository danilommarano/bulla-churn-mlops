"""KFP step: reproduce the train.py stratified train/test split on the prepared parquet."""

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.features.builder import INPUT_COLUMNS


def split_data(
    in_path: str, train_out: str, test_out: str, test_size: float, random_state: int
) -> None:
    """Split the prepared frame into stratified train/test parquets (same split as train.py)."""
    df = pd.read_parquet(in_path)
    X, y = df[INPUT_COLUMNS], df["turnover"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train_df = X_train.copy()
    train_df["turnover"] = y_train.to_numpy()
    test_df = X_test.copy()
    test_df["turnover"] = y_test.to_numpy()

    train_df.to_parquet(train_out, index=False)
    test_df.to_parquet(test_out, index=False)

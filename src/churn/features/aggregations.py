"""Geography churn-rate encoder — learned on TRAIN ONLY (anti-leakage, bug #1)."""

import pandas as pd


class GeographyChurnRateEncoder:
    """Learns the mean `turnover` rate per `Geography` on the training set and applies it.

    - `fit(df_train)`: memorizes the per-group target mean and the global rate (fallback).
    - `transform(df)`: maps each row to the learned rate; unseen geography -> global rate.

    Because the mapping comes exclusively from training data, the test/production set never
    influences the feature value — there is no target leakage.
    """

    def __init__(self, geography_col: str = "Geography", target_col: str = "turnover"):
        self.geography_col = geography_col
        self.target_col = target_col

    def fit(self, df: pd.DataFrame) -> "GeographyChurnRateEncoder":
        self.mapping_ = df.groupby(self.geography_col)[self.target_col].mean().to_dict()
        self.global_rate_ = float(df[self.target_col].mean())
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        if not hasattr(self, "mapping_"):
            raise RuntimeError("Call fit() before transform().")
        return df[self.geography_col].map(self.mapping_).fillna(self.global_rate_).astype(float)

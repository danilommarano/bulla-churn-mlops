"""Leakage-free feature engineering as a scikit-learn transformer.

Learns everything that "sees" the data (age-bucket edges, geography churn rate)
on the TRAIN split only; deterministic derivations (balance per product) need no
fit. Dropping CustomerId and surname_encoded is done by simply never selecting
them (fixes bugs #3 and #4).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer

from churn.features.aggregations import GeographyChurnRateEncoder

# Raw columns fed into the pipeline (CustomerId, Surname, RowNumber, turnover,
# Complain and IsActiveMember are intentionally excluded — see design §2).
RAW_NUMERIC = [
    "CreditScore",
    "Age",
    "Balance",
    "EstimatedSalary",
    "Tenure",
    "NumOfProducts",
    "HasCrCard",
    "Satisfaction Score",
    "Point Earned",
]
RAW_CATEGORICAL = ["Geography", "Gender", "Card Type"]
INPUT_COLUMNS = RAW_NUMERIC + RAW_CATEGORICAL


class ChurnFeatureBuilder(BaseEstimator, TransformerMixin):
    """Adds the derived model features to the raw frame, learning nothing from the target.

    Input: a DataFrame with the INPUT_COLUMNS raw columns. `fit(X, y)` requires the
    target `y` (used only to learn the per-geography churn rate on train) and learns the
    Age quantile-bin edges. `transform(X)` returns X plus three columns:
    `balance_per_product` (Balance / NumOfProducts; Balance nulls filled with 0),
    `geography_churn_rate` (train-learned rate, unseen geography -> global rate) and
    `age_bucket` (ordinal quantile bin of Age). `EstimatedSalary` nulls are filled with 0
    so the downstream StandardScaler never sees NaN.
    """

    def __init__(self, n_age_bins: int = 5, random_state: int = 42):
        self.n_age_bins = n_age_bins
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None) -> "ChurnFeatureBuilder":
        if y is None:
            raise ValueError("ChurnFeatureBuilder.fit requires y (the target).")
        geo_frame = pd.DataFrame(
            {"Geography": X["Geography"].to_numpy(), "turnover": np.asarray(y)}
        )
        self.geo_ = GeographyChurnRateEncoder().fit(geo_frame)
        self.age_ = KBinsDiscretizer(
            n_bins=self.n_age_bins,
            encode="ordinal",
            strategy="quantile",
            quantile_method="averaged_inverted_cdf",
        )
        self.age_.fit(X[["Age"]])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "geo_"):
            raise RuntimeError("Call fit() before transform().")
        X = X.copy()
        X["Balance"] = X["Balance"].fillna(0.0)
        X["EstimatedSalary"] = X["EstimatedSalary"].fillna(0.0)
        X["balance_per_product"] = X["Balance"] / X["NumOfProducts"]
        X["geography_churn_rate"] = self.geo_.transform(X[["Geography"]]).to_numpy()
        X["age_bucket"] = self.age_.transform(X[["Age"]]).ravel()
        return X

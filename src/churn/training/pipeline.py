"""Assembles the single, persistable churn estimator (features + preprocessing + model)."""

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.features.builder import RAW_CATEGORICAL, ChurnFeatureBuilder

# Columns standard-scaled (continuous). balance_per_product is produced by the builder.
SCALE_COLUMNS = [
    "CreditScore",
    "Age",
    "Balance",
    "EstimatedSalary",
    "Point Earned",
    "balance_per_product",
]
# Already-bounded / engineered columns that need no scaling or encoding.
PASSTHROUGH_COLUMNS = [
    "Tenure",
    "NumOfProducts",
    "HasCrCard",
    "Satisfaction Score",
    "geography_churn_rate",
    "age_bucket",
]


def build_pipeline(random_state: int = 42, n_age_bins: int = 5) -> Pipeline:
    """Return the full unfitted churn pipeline."""
    preprocess = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), SCALE_COLUMNS),
            ("onehot", OneHotEncoder(handle_unknown="ignore"), RAW_CATEGORICAL),
            ("pass", "passthrough", PASSTHROUGH_COLUMNS),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            (
                "features",
                ChurnFeatureBuilder(n_age_bins=n_age_bins, random_state=random_state),
            ),
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(
                    max_iter=500, class_weight="balanced", random_state=random_state
                ),
            ),
        ]
    )

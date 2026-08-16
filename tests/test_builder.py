import numpy as np
import pandas as pd
import pytest

from churn.features.builder import INPUT_COLUMNS, ChurnFeatureBuilder  # noqa: F401


def _raw(n_geo, geos, ages, balances, products, target):
    """Minimal frame carrying every INPUT column the builder reads."""
    base = {
        "CreditScore": [650] * n_geo,
        "Age": ages,
        "Balance": balances,
        "EstimatedSalary": [50000.0] * n_geo,
        "Tenure": [5] * n_geo,
        "NumOfProducts": products,
        "HasCrCard": [1] * n_geo,
        "Satisfaction Score": [3] * n_geo,
        "Point Earned": [500] * n_geo,
        "Geography": geos,
        "Gender": ["Male"] * n_geo,
        "Card Type": ["GOLD"] * n_geo,
    }
    return pd.DataFrame(base), pd.Series(target)


def test_transform_adds_derived_columns():
    X, y = _raw(4, ["Sao Paulo"] * 4, [30, 40, 50, 60], [1000.0, 2000.0, 0.0, 4000.0],
                [1, 2, 1, 4], [1, 0, 0, 1])
    out = ChurnFeatureBuilder().fit(X, y).transform(X)
    for col in ("balance_per_product", "geography_churn_rate", "age_bucket"):
        assert col in out.columns


def test_balance_per_product_and_null_fill():
    X, y = _raw(2, ["Sao Paulo", "Sao Paulo"], [30, 40], [np.nan, 2000.0], [2, 4], [1, 0])
    out = ChurnFeatureBuilder().fit(X, y).transform(X)
    # null Balance -> 0 -> 0/2 = 0.0 ; 2000/4 = 500.0
    assert list(out["balance_per_product"]) == [0.0, 500.0]


def test_geography_rate_is_learned_on_train_only():
    # train: SP churn rate 0.5 ; test carries different labels that must NOT leak
    Xtr, ytr = _raw(2, ["Sao Paulo", "Sao Paulo"], [30, 40], [100.0, 200.0], [1, 1], [1, 0])
    Xte, _ = _raw(2, ["Sao Paulo", "Sao Paulo"], [30, 40], [100.0, 200.0], [1, 1], [1, 1])
    out = ChurnFeatureBuilder().fit(Xtr, ytr).transform(Xte)
    assert list(out["geography_churn_rate"]) == [0.5, 0.5]


def test_age_bucket_in_range():
    ages = list(range(20, 40))
    X, y = _raw(20, ["Sao Paulo"] * 20, ages, [1000.0] * 20, [1] * 20, [i % 2 for i in range(20)])
    out = ChurnFeatureBuilder(n_age_bins=5).fit(X, y).transform(X)
    assert out["age_bucket"].min() >= 0
    assert out["age_bucket"].max() <= 4


def test_transform_before_fit_raises():
    X, _ = _raw(1, ["Sao Paulo"], [30], [1000.0], [1], [0])
    with pytest.raises(RuntimeError):
        ChurnFeatureBuilder().transform(X)

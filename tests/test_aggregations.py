import pandas as pd
import pytest

from churn.features.aggregations import GeographyChurnRateEncoder


def _frame(geos, targets) -> pd.DataFrame:
    return pd.DataFrame({"Geography": geos, "turnover": targets})


def test_fit_learns_train_group_means():
    train = _frame(
        ["Sao Paulo", "Sao Paulo", "Minas Gerais", "Minas Gerais"],
        [1, 0, 0, 0],  # SP: 0.5, MG: 0.0
    )
    enc = GeographyChurnRateEncoder().fit(train)
    assert enc.mapping_["Sao Paulo"] == 0.5
    assert enc.mapping_["Minas Gerais"] == 0.0


def test_transform_uses_train_mapping_not_test_labels():
    # In train, SP has churn 0.5. In test, SP has churn 1.0.
    # transform MUST return 0.5 (learned on train), never 1.0 (the test label).
    train = _frame(["Sao Paulo", "Sao Paulo"], [1, 0])  # SP: 0.5
    test = _frame(["Sao Paulo", "Sao Paulo"], [1, 1])   # SP: 1.0 (must not leak)
    enc = GeographyChurnRateEncoder().fit(train)
    out = enc.transform(test)
    assert list(out) == [0.5, 0.5]


def test_unseen_geography_falls_back_to_global_rate():
    train = _frame(["Sao Paulo", "Sao Paulo", "Minas Gerais"], [1, 0, 0])  # global = 1/3
    enc = GeographyChurnRateEncoder().fit(train)
    out = enc.transform(_frame(["Rio de Janeiro"], [0]))
    assert abs(out.iloc[0] - (1 / 3)) < 1e-9


def test_transform_before_fit_raises():
    enc = GeographyChurnRateEncoder()
    with pytest.raises(RuntimeError):
        enc.transform(_frame(["Sao Paulo"], [0]))


def test_fit_returns_self():
    train = _frame(["Sao Paulo", "Minas Gerais"], [1, 0])
    enc = GeographyChurnRateEncoder()
    assert enc.fit(train) is enc

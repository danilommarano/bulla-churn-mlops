import pandas as pd

from churn.monitoring.perturb import simulate_drift


def _frame():
    return pd.DataFrame(
        {
            "Age": [30, 40, 50],
            "Balance": [1000.0, 2000.0, 3000.0],
            "CreditScore": [600, 650, 700],
            "Geography": ["France", "Spain", "Germany"],
            "turnover": [0, 1, 0],
        }
    )


def test_simulate_drift_is_deterministic():
    a = simulate_drift(_frame())
    b = simulate_drift(_frame())
    pd.testing.assert_frame_equal(a, b)


def test_simulate_drift_shifts_a_numeric_mean():
    original = _frame()
    drifted = simulate_drift(original)
    assert drifted["Age"].mean() != original["Age"].mean()


def test_simulate_drift_preserves_schema_and_target():
    original = _frame()
    drifted = simulate_drift(original)
    assert list(drifted.columns) == list(original.columns)
    assert drifted["turnover"].tolist() == original["turnover"].tolist()
    # does not mutate the input
    assert original["Age"].mean() == _frame()["Age"].mean()

import numpy as np

from churn.scoring import retention_score


def test_scalar_bounds():
    assert retention_score(0.0) == 0  # no chance of staying -> score 0
    assert retention_score(1.0) == 10  # certain to stay -> score 10


def test_scalar_rounding():
    assert retention_score(0.42) == 4  # 4.2 → 4
    assert retention_score(0.78) == 8  # 7.8 → 8


def test_array_input():
    out = retention_score(np.array([0.0, 0.42, 1.0]))
    assert out.tolist() == [0, 4, 10]
    assert out.dtype == np.int64 or out.dtype == int

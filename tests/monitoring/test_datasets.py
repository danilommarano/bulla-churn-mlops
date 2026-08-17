# tests/monitoring/test_datasets.py
import numpy as np

from churn.config import Settings
from churn.features.builder import INPUT_COLUMNS
from churn.monitoring.datasets import build_reference_current


class _FakeModel:
    """Returns a deterministic 2-column proba based on row position."""

    def predict_proba(self, frame):
        n = len(frame)
        churn = np.linspace(0.1, 0.9, n)
        return np.column_stack([1 - churn, churn])


def test_build_reference_current_shapes_and_columns(tmp_path):
    cfg = Settings()  # uses real Customer-Churn-Records.csv at repo root
    reference, current = build_reference_current(cfg, model=_FakeModel())

    for frame in (reference, current):
        for col in INPUT_COLUMNS:
            assert col in frame.columns
        assert "prob_churn" in frame.columns
        assert "turnover" in frame.columns
        assert frame["prob_churn"].between(0.0, 1.0).all()

    # current is the holdout: test_size fraction of the full dataset
    total = len(reference) + len(current)
    assert abs(len(current) / total - cfg.test_size) < 0.01


def test_build_reference_current_can_perturb(tmp_path):
    cfg = Settings()
    _, current_plain = build_reference_current(cfg, model=_FakeModel())
    _, current_drift = build_reference_current(
        cfg, model=_FakeModel(), simulate=True
    )
    assert current_drift["Age"].mean() != current_plain["Age"].mean()

# tests/monitoring/test_report.py
import numpy as np
import pandas as pd

from churn.config import Settings
from churn.monitoring.report import build_report


def _scored_frame(offset: float = 0.0):
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "CreditScore": rng.normal(650 + offset * 100, 50, n),
            "Age": rng.normal(40 + offset * 10, 8, n),
            "Balance": rng.normal(60000, 10000, n),
            "EstimatedSalary": rng.normal(100000, 20000, n),
            "Tenure": rng.integers(0, 10, n),
            "NumOfProducts": rng.integers(1, 4, n),
            "HasCrCard": rng.integers(0, 2, n),
            "Satisfaction Score": rng.integers(1, 6, n),
            "Point Earned": rng.normal(600, 100, n),
            "Geography": rng.choice(["France", "Spain", "Germany"], n),
            "Gender": rng.choice(["Male", "Female"], n),
            "Card Type": rng.choice(["SILVER", "GOLD", "PLATINUM", "DIAMOND"], n),
            "prob_churn": rng.uniform(0, 1, n),
            "turnover": rng.integers(0, 2, n),
        }
    )


def test_build_report_generates_html(tmp_path):
    reference = _scored_frame(0.0)
    current = _scored_frame(0.0)
    snapshot = build_report(reference, current, Settings())
    out = tmp_path / "drift.html"
    snapshot.save_html(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_build_report_snapshot_dict_has_metrics():
    reference = _scored_frame(0.0)
    current = _scored_frame(0.0)
    snapshot = build_report(reference, current, Settings())
    payload = snapshot.dict()
    assert "metrics" in payload

# tests/monitoring/test_cli.py
import json

from churn.config import Settings
from churn.monitoring.__main__ import run


class _FakeModel:
    import numpy as _np

    def predict_proba(self, frame):
        import numpy as np

        n = len(frame)
        churn = np.linspace(0.1, 0.9, n)
        return np.column_stack([1 - churn, churn])


def test_run_writes_outputs_and_returns_exit_code(tmp_path, monkeypatch):
    cfg = Settings(reports_dir=str(tmp_path))
    monkeypatch.setattr(
        "churn.monitoring.__main__.load_production_model", lambda c: _FakeModel()
    )
    exit_code = run(cfg, simulate=False)

    assert (tmp_path / "drift.html").exists()
    metrics_path = tmp_path / "metrics.json"
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text())
    assert "drift_share" in payload
    assert exit_code in (0, 1)

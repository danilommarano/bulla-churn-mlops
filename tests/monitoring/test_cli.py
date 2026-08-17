# tests/monitoring/test_cli.py
import json

from churn.config import Settings
from churn.monitoring.__main__ import run


class _FakeModel:
    def predict_proba(self, frame):
        import numpy as np

        n = len(frame)
        churn = np.linspace(0.1, 0.9, n)
        return np.column_stack([1 - churn, churn])


def _patched_cfg(tmp_path, monkeypatch):
    cfg = Settings(reports_dir=str(tmp_path))
    monkeypatch.setattr(
        "churn.monitoring.__main__.load_production_model", lambda c: _FakeModel()
    )
    return cfg


def test_run_healthy_writes_outputs_and_passes_gate(tmp_path, monkeypatch):
    cfg = _patched_cfg(tmp_path, monkeypatch)
    exit_code = run(cfg, simulate=False)

    assert (tmp_path / "drift.html").exists()
    metrics_path = tmp_path / "metrics.json"
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text())
    assert "drift_share" in payload
    assert "quality" in payload
    # Holdout unperturbed -> no drift -> gate passes.
    assert exit_code == 0


def test_run_simulated_drift_fails_gate(tmp_path, monkeypatch):
    cfg = _patched_cfg(tmp_path, monkeypatch)
    # The headline behavior: --simulate-drift must trip the gate (exit 1), the
    # local stand-in for a Vertex AI Model Monitoring alert.
    assert run(cfg, simulate=True) == 1

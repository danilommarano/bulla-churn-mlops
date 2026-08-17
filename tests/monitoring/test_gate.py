# tests/monitoring/test_gate.py
from churn.config import Settings
from churn.monitoring.gate import evaluate_gate, summarize
from churn.monitoring.report import build_report
from tests.monitoring.test_report import _scored_frame


def test_evaluate_gate_passes_when_share_within_threshold():
    assert evaluate_gate({"drift_share": 0.2}, threshold=0.3) is True


def test_evaluate_gate_fails_when_share_exceeds_threshold():
    assert evaluate_gate({"drift_share": 0.5}, threshold=0.3) is False


def test_summarize_extracts_drift_share_from_real_snapshot():
    snapshot = build_report(_scored_frame(0.0), _scored_frame(0.0), Settings())
    summary = summarize(snapshot)
    assert "drift_share" in summary
    assert 0.0 <= summary["drift_share"] <= 1.0

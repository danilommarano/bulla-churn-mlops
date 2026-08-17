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
    # Reference vs a SHIFTED current, so real drift is present. This pins that
    # _find_metric actually located DriftedColumnsCount and parsed its share —
    # an identical (0.0 vs 0.0) pair would pass even if parsing silently returned 0.
    snapshot = build_report(_scored_frame(0.0), _scored_frame(1.0), Settings())
    summary = summarize(snapshot)
    assert summary["drift_share"] > 0.0
    assert summary["drifted_columns"] >= 1


def test_summarize_surfaces_model_quality_metrics():
    snapshot = build_report(_scored_frame(0.0), _scored_frame(0.0), Settings())
    quality = summarize(snapshot)["quality"]
    assert "roc_auc" in quality
    assert 0.0 <= quality["roc_auc"] <= 1.0

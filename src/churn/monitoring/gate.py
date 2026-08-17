"""Summarize an Evidently snapshot and apply the drift quality gate.

The gate is the local stand-in for Vertex AI Model Monitoring's managed alert:
if the share of drifted input features exceeds the threshold, the run fails.
"""


def _find_metric(metrics, needle: str):
    """Return the first metric dict whose display name / config type contains needle.

    Evidently 0.7 metric dicts have keys ``id`` (an opaque hash), ``metric_name``
    (a display string like ``DriftedColumnsCount(...)``), ``config`` (a dict with a
    ``type`` like ``evidently:metric_v2:DriftedColumnsCount``), and ``value``. Match
    on ``metric_name`` or ``config["type"]`` since ``id`` is not human-meaningful.
    """
    needle_lower = needle.lower()
    for metric in metrics:
        name = str(metric.get("metric_name", ""))
        config = metric.get("config") or {}
        config_type = str(config.get("type", "")) if isinstance(config, dict) else ""
        if needle_lower in name.lower() or needle_lower in config_type.lower():
            return metric
    return None


def summarize(snapshot) -> dict:
    """Extract the key monitoring numbers into a JSON-serializable dict."""
    payload = snapshot.dict()
    metrics = payload.get("metrics", [])
    drift = _find_metric(metrics, "DriftedColumnsCount") or {}
    value = drift.get("value", {})
    # value is expected like {"count": N, "share": S}; fall back defensively.
    if isinstance(value, dict):
        drift_share = float(value.get("share", 0.0))
        drifted_count = int(value.get("count", 0))
    else:
        drift_share, drifted_count = 0.0, 0
    return {"drift_share": drift_share, "drifted_columns": drifted_count}


def evaluate_gate(summary: dict, threshold: float) -> bool:
    """Return True (pass) when drift share is within threshold, else False (alert)."""
    return summary.get("drift_share", 0.0) <= threshold

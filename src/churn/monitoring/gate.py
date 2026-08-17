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


# Evidently metric_name -> summary key for the model-quality half of the report
# (Vertex AI "model quality"). Each has a scalar float value in snapshot.dict().
_QUALITY_METRICS = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1Score": "f1",
    "RocAuc": "roc_auc",
    "LogLoss": "log_loss",
}


def summarize(snapshot) -> dict:
    """Extract the key monitoring numbers into a JSON-serializable dict.

    Raises RuntimeError if the drift metric is missing or malformed: for a
    monitoring tool the safe failure mode is a loud error, not a silent "no drift"
    that would let a broken run exit 0 and look healthy.
    """
    payload = snapshot.dict()
    metrics = payload.get("metrics", [])

    drift = _find_metric(metrics, "DriftedColumnsCount")
    if drift is None:
        raise RuntimeError(
            "DriftedColumnsCount metric not found in the Evidently snapshot; "
            "the Evidently API may have changed (see report.py / gate.py)."
        )
    value = drift.get("value")
    if not isinstance(value, dict) or "share" not in value or "count" not in value:
        raise RuntimeError(f"Unexpected DriftedColumnsCount value shape: {value!r}")

    quality = {}
    for name, key in _QUALITY_METRICS.items():
        metric = _find_metric(metrics, name)
        if metric is not None and not isinstance(metric.get("value"), dict):
            quality[key] = float(metric["value"])

    return {
        "drift_share": float(value["share"]),
        "drifted_columns": int(value["count"]),
        "quality": quality,
    }


def evaluate_gate(summary: dict, threshold: float) -> bool:
    """Return True (pass) when drift share is within threshold, else False (alert)."""
    return summary.get("drift_share", 0.0) <= threshold

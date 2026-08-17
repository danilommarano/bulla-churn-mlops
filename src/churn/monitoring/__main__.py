# src/churn/monitoring/__main__.py
"""CLI entrypoint: run a monitoring pass, write reports, apply the quality gate.

    python -m churn.monitoring [--simulate-drift]

Exit code 0 = gate passed (drift within threshold), 1 = gate failed (alert).
"""

import argparse
import json
import sys
from pathlib import Path

from churn.config import Settings, settings
from churn.monitoring.datasets import build_reference_current
from churn.monitoring.gate import evaluate_gate, summarize
from churn.monitoring.report import build_report
from churn.serving.api import load_production_model


def run(cfg: Settings = settings, *, simulate: bool = False) -> int:
    """Execute one monitoring pass. Returns the process exit code (0 pass, 1 alert)."""
    model = load_production_model(cfg)
    reference, current = build_reference_current(cfg, model=model, simulate=simulate)
    snapshot = build_report(reference, current, cfg)

    Path(cfg.reports_dir).mkdir(parents=True, exist_ok=True)
    snapshot.save_html(cfg.monitoring_report_path)
    summary = summarize(snapshot)
    Path(cfg.monitoring_metrics_path).write_text(json.dumps(summary, indent=2))

    passed = evaluate_gate(summary, cfg.drift_threshold)
    status = "PASS" if passed else "ALERT"
    print(
        f"[{status}] drift_share={summary['drift_share']:.3f} "
        f"threshold={cfg.drift_threshold} drifted_columns={summary['drifted_columns']}"
    )
    print(
        f"report: {cfg.monitoring_report_path}  metrics: {cfg.monitoring_metrics_path}"
    )
    return 0 if passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Churn model monitoring (drift + quality)."
    )
    parser.add_argument(
        "--simulate-drift",
        action="store_true",
        help="Perturb the holdout to demonstrate drift detection.",
    )
    args = parser.parse_args()
    sys.exit(run(settings, simulate=args.simulate_drift))


if __name__ == "__main__":
    main()

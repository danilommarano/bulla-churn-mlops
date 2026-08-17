# Model Monitoring com Evidently — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Evidently-based model-monitoring CLI (Vertex AI Model Monitoring equivalent) that reports input/score drift + model quality and fails a quality gate (non-zero exit) when drift exceeds a threshold.

**Architecture:** New package `src/churn/monitoring/` with pure functions over paths/frames (`datasets`, `perturb`, `report`, `gate`) plus a thin `__main__` CLI. Reference = train split, current = holdout (optionally perturbed). Outputs `reports/drift.html` + `reports/metrics.json`. Mirrors the existing `feature_store/` and `orchestration/` layout.

**Tech Stack:** Python 3.12, evidently 0.7.21, scikit-learn, pandas 2.x, mlflow (production model), pytest, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-08-17-model-monitoring.md`

---

## Context for the implementer

- The training split (must be reproduced exactly for the reference set):
  ```python
  X, y = df[INPUT_COLUMNS], df["turnover"]
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
  )
  ```
- `INPUT_COLUMNS = RAW_NUMERIC + RAW_CATEGORICAL` from `churn.features.builder`
  (9 numeric + 3 categorical = 12 raw columns). Target column is `turnover`.
- The production model is a full sklearn `Pipeline` (`ChurnFeatureBuilder` derives
  its own features internally) loaded via
  `churn.serving.api.load_production_model(cfg)` →
  `models:/churn-model@production`. `model.predict_proba(frame)[:, 1]` = prob churn.
- `Settings` (`churn.config`) uses `env_prefix="CHURN_"`, has `random_state=42`,
  `test_size=0.2`, and derived-path `@property` accessors (see `feast_offline_path`).
- Existing test suite: 62 tests green. `make test` = `uv run pytest -v`.
  `make lint` = `uv run ruff check src tests`.
- `reports/` and `*.html` are already gitignored. Never `git add` generated HTML/JSON.
- Repo policy: conventional-commit messages in English, **no** `Co-Authored-By: Claude`
  trailer in this repo.

---

## Task 1: Config additions

**Files:**
- Modify: `src/churn/config.py`
- Test: `tests/test_config.py` (create if missing)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py  (add these; create file with imports if it doesn't exist)
from churn.config import Settings


def test_monitoring_defaults_and_paths():
    cfg = Settings()
    assert cfg.reports_dir == "reports"
    assert cfg.drift_threshold == 0.3
    assert cfg.monitoring_report_path == "reports/drift.html"
    assert cfg.monitoring_metrics_path == "reports/metrics.json"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_monitoring_defaults_and_paths -v`
Expected: FAIL (AttributeError: 'Settings' object has no attribute 'reports_dir')

- [ ] **Step 3: Implement**

In `src/churn/config.py`, after the Feast block and before `settings = Settings()`,
add the fields and properties (keep `Path` import — already present):

```python
    # Model monitoring (Vertex AI Model Monitoring equivalent)
    reports_dir: str = "reports"
    drift_threshold: float = 0.3  # Jensen-Shannon per-feature, Vertex default parity

    @property
    def monitoring_report_path(self) -> str:
        return str(Path(self.reports_dir) / "drift.html")

    @property
    def monitoring_metrics_path(self) -> str:
        return str(Path(self.reports_dir) / "metrics.json")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/churn/config.py tests/test_config.py
git commit -m "feat: add monitoring config (reports_dir, drift_threshold, report paths)"
```

---

## Task 2: `perturb.py` — deterministic drift simulation

**Files:**
- Create: `src/churn/monitoring/__init__.py` (empty)
- Create: `src/churn/monitoring/perturb.py`
- Test: `tests/monitoring/test_perturb.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/monitoring/test_perturb.py
import pandas as pd

from churn.monitoring.perturb import simulate_drift


def _frame():
    return pd.DataFrame(
        {
            "Age": [30, 40, 50],
            "Balance": [1000.0, 2000.0, 3000.0],
            "CreditScore": [600, 650, 700],
            "Geography": ["France", "Spain", "Germany"],
            "turnover": [0, 1, 0],
        }
    )


def test_simulate_drift_is_deterministic():
    a = simulate_drift(_frame())
    b = simulate_drift(_frame())
    pd.testing.assert_frame_equal(a, b)


def test_simulate_drift_shifts_a_numeric_mean():
    original = _frame()
    drifted = simulate_drift(original)
    assert drifted["Age"].mean() != original["Age"].mean()


def test_simulate_drift_preserves_schema_and_target():
    original = _frame()
    drifted = simulate_drift(original)
    assert list(drifted.columns) == list(original.columns)
    assert drifted["turnover"].tolist() == original["turnover"].tolist()
    # does not mutate the input
    assert original["Age"].mean() == _frame()["Age"].mean()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/monitoring/test_perturb.py -v`
Expected: FAIL (ModuleNotFoundError: churn.monitoring.perturb)

- [ ] **Step 3: Implement**

Create `src/churn/monitoring/__init__.py` (empty file).

```python
# src/churn/monitoring/perturb.py
"""Deterministic drift simulation for demonstrating monitoring detection.

Shifts a few numeric feature distributions by fixed amounts so a monitoring run
on the perturbed holdout produces visible input + score drift. Categorical
columns and the target are preserved. No global RNG; fully deterministic.
"""

import pandas as pd

# Fixed additive/multiplicative shifts applied to raw numeric features present.
_SHIFTS = {
    "Age": lambda s: s + 15,
    "Balance": lambda s: s * 1.5,
    "CreditScore": lambda s: s - 100,
    "EstimatedSalary": lambda s: s * 1.3,
}


def simulate_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with selected numeric columns shifted. Pure/deterministic."""
    out = df.copy()
    for column, shift in _SHIFTS.items():
        if column in out.columns:
            out[column] = shift(out[column])
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/monitoring/test_perturb.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/churn/monitoring/__init__.py src/churn/monitoring/perturb.py tests/monitoring/test_perturb.py
git commit -m "feat: add deterministic drift simulation for monitoring"
```

---

## Task 3: `datasets.py` — build reference & current frames

**Files:**
- Create: `src/churn/monitoring/datasets.py`
- Test: `tests/monitoring/test_datasets.py`

The reference/current frames each carry the 12 raw columns + `prob_churn` (model
score) + `turnover` (label). The model is injected as a parameter so tests can pass
a fake predictor (no MLflow dependency in the unit test).

- [ ] **Step 1: Write the failing test**

```python
# tests/monitoring/test_datasets.py
import numpy as np
import pandas as pd

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/monitoring/test_datasets.py -v`
Expected: FAIL (ModuleNotFoundError: churn.monitoring.datasets)

- [ ] **Step 3: Implement**

```python
# src/churn/monitoring/datasets.py
"""Build the reference and current datasets for a monitoring run.

Reference = train split (baseline the model was fit on). Current = holdout
(X_test), optionally perturbed with simulate_drift to demonstrate detection.
Each frame carries the 12 raw INPUT_COLUMNS + prob_churn (model score) + turnover.
The split reproduces training exactly (same random_state / test_size / stratify).
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import Settings, settings
from churn.data import load_raw
from churn.features.builder import INPUT_COLUMNS
from churn.monitoring.perturb import simulate_drift


def _scored(X: pd.DataFrame, y: pd.Series, model) -> pd.DataFrame:
    frame = X[INPUT_COLUMNS].copy()
    frame["prob_churn"] = model.predict_proba(frame[INPUT_COLUMNS])[:, 1]
    frame["turnover"] = y.to_numpy()
    return frame


def build_reference_current(
    cfg: Settings = settings, *, model, simulate: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (reference, current) scored frames for a monitoring run."""
    df = load_raw(cfg.data_path)
    X, y = df[INPUT_COLUMNS], df["turnover"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    if simulate:
        X_test = simulate_drift(X_test)
    reference = _scored(X_train, y_train, model)
    current = _scored(X_test, y_test, model)
    return reference, current
```

> Note: `simulate_drift` shifts raw feature columns; the perturbed `X_test` is then
> scored, so both input drift and score drift appear. `X_test` retains `INPUT_COLUMNS`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/monitoring/test_datasets.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/churn/monitoring/datasets.py tests/monitoring/test_datasets.py
git commit -m "feat: build reference/current scored datasets for monitoring"
```

---

## Task 4: `report.py` — Evidently report (probe API first)

**Files:**
- Create: `src/churn/monitoring/report.py`
- Test: `tests/monitoring/test_report.py`

Evidently 0.7's exact import paths/signatures for `ValueDrift`,
`ClassificationQuality`, `BinaryClassification`, and the drift-method/threshold
config are version-sensitive. **Probe the real API before writing the module.**

- [ ] **Step 0: Probe the installed Evidently API**

Run these and record the actual working imports/signatures:

```bash
uv run python - <<'PY'
import evidently, inspect
print("evidently", evidently.__version__)
import evidently as E
for name in ["Report", "Dataset", "DataDefinition", "BinaryClassification", "Regression"]:
    print("evidently.", name, hasattr(E, name))
import evidently.presets as P
print("presets:", [n for n in dir(P) if n[0].isupper()])
import evidently.metrics as M
print("metrics has ValueDrift:", hasattr(M, "ValueDrift"))
print(inspect.signature(E.Report.__init__))
PY
```

Confirm: `Report`, `Dataset`, `DataDefinition`, `BinaryClassification` importable
from `evidently`; locate `DataDriftPreset` + `ClassificationQuality`/classification
preset in `evidently.presets`; locate `ValueDrift` (likely `evidently.metrics`).
Adjust the imports in Step 3 to match what the probe prints. Also confirm whether
`DataDriftPreset` accepts a per-column drift method / threshold argument; if it
does, set Jensen-Shannon + `cfg.drift_threshold`; if the surface differs, keep the
default preset and record the deviation in a code comment (the gate still enforces
the threshold via drift share).

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/monitoring/test_report.py -v`
Expected: FAIL (ModuleNotFoundError: churn.monitoring.report)

- [ ] **Step 3: Implement (adjust imports to the probe output)**

```python
# src/churn/monitoring/report.py
"""Build the Evidently monitoring report (input drift + score drift + quality).

Uses the Evidently 0.7 API. Data/score drift use Jensen-Shannon (Vertex AI Model
Monitoring parity); the per-feature threshold comes from cfg.drift_threshold.
"""

from evidently import BinaryClassification, Dataset, DataDefinition, Report
from evidently.metrics import ValueDrift
from evidently.presets import ClassificationQuality, DataDriftPreset

from churn.config import Settings
from churn.features.builder import RAW_CATEGORICAL, RAW_NUMERIC


def _dataset(frame, definition):
    return Dataset.from_pandas(frame, data_definition=definition)


def build_report(reference_df, current_df, cfg: Settings):
    """Run the monitoring report and return the Evidently snapshot."""
    definition = DataDefinition(
        numerical_columns=[*RAW_NUMERIC, "prob_churn"],
        categorical_columns=list(RAW_CATEGORICAL),
        classification=[
            BinaryClassification(
                target="turnover", prediction_probas="prob_churn", pos_label=1
            )
        ],
    )
    report = Report(
        [
            DataDriftPreset(),
            ValueDrift(column="prob_churn"),
            ClassificationQuality(),
        ],
        include_tests=True,
    )
    reference = _dataset(reference_df, definition)
    current = _dataset(current_df, definition)
    return report.run(current_data=current, reference_data=reference)
```

> If the probe showed `classification` takes a single object (not a list), or
> `ClassificationQuality`/`ValueDrift` live elsewhere, adjust accordingly. Keep the
> keyword `current_data=`/`reference_data=` call form.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/monitoring/test_report.py -v`
Expected: PASS (2 passed). If a preset rejects the data, narrow the metric list to
what the probe validated and note it in a comment — the DataDriftPreset + ValueDrift
are the must-haves.

- [ ] **Step 5: Commit**

```bash
git add src/churn/monitoring/report.py tests/monitoring/test_report.py
git commit -m "feat: build Evidently drift + quality report for monitoring"
```

---

## Task 5: `gate.py` — summarize snapshot + quality gate

**Files:**
- Create: `src/churn/monitoring/gate.py`
- Test: `tests/monitoring/test_gate.py`

`evaluate_gate` is pure and unit-tested directly. `summarize` is tested against a
real snapshot built via `build_report` so the parsing matches Evidently's actual
`snapshot.dict()` structure.

- [ ] **Step 0: Discover the snapshot dict structure**

Run to see where the drifted-column share lives in `snapshot.dict()`:

```bash
uv run python - <<'PY'
import json, numpy as np, pandas as pd
from churn.config import Settings
from churn.monitoring.report import build_report
from tests.monitoring.test_report import _scored_frame
snap = build_report(_scored_frame(0.0), _scored_frame(1.0), Settings())
print(json.dumps(snap.dict(), indent=2, default=str)[:4000])
PY
```

Identify the metric id/label for the drifted-columns count/share (e.g.
`DriftedColumnsCount`) and the exact keys holding the share and the count. Use those
keys in `summarize`. Prefer a defensive lookup (search the metrics list by
metric id substring) over a brittle fixed index.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/monitoring/test_gate.py -v`
Expected: FAIL (ModuleNotFoundError: churn.monitoring.gate)

- [ ] **Step 3: Implement (adjust key lookup to the Step 0 output)**

```python
# src/churn/monitoring/gate.py
"""Summarize an Evidently snapshot and apply the drift quality gate.

The gate is the local stand-in for Vertex AI Model Monitoring's managed alert:
if the share of drifted input features exceeds the threshold, the run fails.
"""


def _find_metric(metrics, needle: str):
    """Return the first metric dict whose id/metric_id contains needle (or None)."""
    for metric in metrics:
        identifier = str(metric.get("metric_id") or metric.get("id") or metric)
        if needle.lower() in identifier.lower():
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
```

> Adjust `summarize` to the real keys the Step 0 probe printed. The two contract
> points the tests pin: the output dict has `drift_share` in [0,1], and
> `evaluate_gate` compares it to the threshold.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/monitoring/test_gate.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/churn/monitoring/gate.py tests/monitoring/test_gate.py
git commit -m "feat: add monitoring snapshot summary and drift quality gate"
```

---

## Task 6: `__main__.py` — CLI wiring + exit code

**Files:**
- Create: `src/churn/monitoring/__main__.py`
- Test: `tests/monitoring/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/monitoring/test_cli.py -v`
Expected: FAIL (ModuleNotFoundError / cannot import run)

- [ ] **Step 3: Implement**

```python
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
    print(f"report: {cfg.monitoring_report_path}  metrics: {cfg.monitoring_metrics_path}")
    return 0 if passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Churn model monitoring (drift + quality).")
    parser.add_argument(
        "--simulate-drift",
        action="store_true",
        help="Perturb the holdout to demonstrate drift detection.",
    )
    args = parser.parse_args()
    sys.exit(run(settings, simulate=args.simulate_drift))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/monitoring/test_cli.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/churn/monitoring/__main__.py tests/monitoring/test_cli.py
git commit -m "feat: add monitoring CLI with drift quality-gate exit code"
```

---

## Task 7: Makefile targets + full verification

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add targets**

Add `monitor monitor-drift` to the `.PHONY` line, and append:

```makefile
monitor: ## Gera o relatório de drift/qualidade (holdout saudável; gate deve passar)
	uv run python -m churn.monitoring

monitor-drift: ## Idem com drift simulado (demonstra detecção; gate falha de propósito)
	uv run python -m churn.monitoring --simulate-drift
```

- [ ] **Step 2: Verify the full suite + lint**

Run: `uv run pytest -v`
Expected: all tests pass (62 prior + new).

Run: `uv run ruff check src tests`
Expected: clean.

- [ ] **Step 3: End-to-end smoke (requires a trained model)**

Ensure a model is registered (`make train` if needed), then:

Run: `make monitor`
Expected: prints `[PASS] drift_share=...`, exit 0, `reports/drift.html` + `reports/metrics.json` written.

Run: `make monitor-drift`
Expected: prints `[ALERT] drift_share=...` with higher share, exit 1.

> Note the `$?` exit code manually; `make` will report the non-zero exit for
> `monitor-drift` as an error — that is the intended "alert" behavior. Mention this
> in the PR so it is not mistaken for a failure.

- [ ] **Step 4: Commit**

```bash
git add Makefile
git commit -m "feat: add make monitor and monitor-drift targets"
```

---

## Self-Review notes (author)

- **Spec coverage:** §3 modules → Tasks 2-6; §4 config → Task 1; §5 Makefile →
  Task 7; §6 tests → embedded per task. Mapping table §2 lives in the spec/README,
  not code.
- **API risk:** Tasks 4 and 5 begin with a real-API probe step because Evidently
  0.7's metric imports and `snapshot.dict()` shape are version-sensitive; the
  implementer pins the exact signature before writing, then keeps the tested
  contract (HTML generated; `drift_share` in [0,1]; gate compares to threshold).
- **Type consistency:** `build_reference_current(cfg, *, model, simulate)` →
  `build_report(reference, current, cfg)` → `summarize(snapshot)` /
  `evaluate_gate(summary, threshold)` → `run(cfg, *, simulate)` used consistently
  across tasks and the CLI test.
```

# KFP Pipeline Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Orchestrate the churn training lifecycle as a 5-stage KFP local DAG (`prepare_data → split_data → train_model → evaluate_model → register_model`) with typed artifact lineage, mirroring Vertex AI Pipelines.

**Architecture:** Each stage has a **pure function** in `src/churn/orchestration/steps/<stage>.py` (operates on file paths + primitives, reuses existing `churn` ML code, unit-testable without KFP) and a thin `@dsl.component` **wrapper** in `src/churn/orchestration/dag.py` (maps KFP artifacts → paths, delegates to the pure function). A DRY helper `churn/training/registry.py::log_and_register` centralizes MLflow logging/registration, reused by both `train.py` and the `register_model` step. The `register_model` step promotes the `@production` alias only if `roc_auc >= min_roc_auc` (quality gate). Runs locally via `local.SubprocessRunner(use_venv=False)` — no cluster, no images.

**Tech Stack:** Python 3.12, uv, pytest, ruff, kfp 2.17, MLflow 3.15, scikit-learn, pandas 2.x, joblib.

**Spec:** [`docs/superpowers/specs/2026-08-17-pipeline-orchestration.md`](../specs/2026-08-17-pipeline-orchestration.md)

---

## Verified KFP facts (from Fase A spike + API checks on kfp 2.17.0)

These are load-bearing and already validated against the installed version — trust them:

- `@dsl.component` functions **must live in a real module file** (KFP uses `inspect.getsource`); they cannot be defined in a REPL/heredoc. `dag.py` is a real file, so this is fine.
- In a `@dsl.pipeline`, a task's outputs are keyed by the **Output parameter name**: a component with `output: Output[Dataset]` is consumed as `task.outputs["output"]`.
- `Input/Output[Dataset|Model|Metrics]` expose `.path` — a writable file path that **persists to downstream components**. Writing JSON to an `Output[Metrics].path` and reading it via `Input[Metrics].path` downstream works.
- `Output[Metrics].log_metric(name, value)` records scalar metrics for lineage (separate from `.path`).
- `local.init(runner=local.SubprocessRunner(use_venv=False), raise_on_error=True, enable_caching=False)` — `use_venv=False` runs components in the current venv (so `import churn` works); `raise_on_error=True` raises on component failure; `enable_caching=False` avoids stale reruns.
- Calling the pipeline function returns a run object whose `.state` is `TaskState.FINAL` on success (with `raise_on_error=True`, failures raise instead of returning).
- Inside a `@dsl.component`, imports must be **inside the function body** (KFP executes the function source standalone). Put `from churn... import ...` inside each wrapper.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/churn/config.py` | modify | Add `min_roc_auc: float = 0.70` |
| `src/churn/training/registry.py` | create | DRY `log_and_register(pipeline, X_sample, metrics, cfg, promote)` |
| `src/churn/training/train.py` | modify | Call `log_and_register` instead of inline MLflow block |
| `src/churn/orchestration/__init__.py` | create | Package marker |
| `src/churn/orchestration/steps/__init__.py` | create | Subpackage marker |
| `src/churn/orchestration/steps/prepare_data.py` | create | `prepare_data(out_path, data_path)` |
| `src/churn/orchestration/steps/split_data.py` | create | `split_data(in_path, train_out, test_out, test_size, random_state)` |
| `src/churn/orchestration/steps/train_model.py` | create | `train_model(train_path, model_out, random_state, n_age_bins)` |
| `src/churn/orchestration/steps/evaluate_model.py` | create | `evaluate_model(model_path, test_path, metrics_out)` |
| `src/churn/orchestration/steps/register_model.py` | create | `register_model(model_path, metrics_path, train_path, cfg)` |
| `src/churn/orchestration/dag.py` | create | `@dsl.component` wrappers + `@dsl.pipeline` + `run_local`/`main` |
| `tests/test_orchestration.py` | create | Unit tests per step + e2e DAG test |
| `Makefile` | modify | `pipeline` target |

Reused unchanged: `churn.data.load_raw`, `churn.features.builder` (`INPUT_COLUMNS`, `ChurnFeatureBuilder`), `churn.training.pipeline.build_pipeline`, `churn.training.evaluate.evaluate`.

---

## Task 1: Quality-gate config + DRY `log_and_register` helper

Foundation. Extract the MLflow block from `train.py` into a reusable helper (behavior-preserving) and add the gate threshold to config. The 30 existing training tests are the safety net.

**Files:**
- Modify: `src/churn/config.py`
- Create: `src/churn/training/registry.py`
- Modify: `src/churn/training/train.py`
- Test: `tests/test_registry.py` (create)

- [ ] **Step 1: Add `min_roc_auc` to Settings**

In `src/churn/config.py`, add the field right after `n_age_bins: int = 5` (line 16):

```python
    n_age_bins: int = 5
    min_roc_auc: float = 0.70  # promotion gate: register_model moves @production only if roc_auc >= this
```

- [ ] **Step 2: Write the failing test for `log_and_register`**

Create `tests/test_registry.py`:

```python
from pathlib import Path

import pandas as pd
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split

from churn.config import Settings
from churn.data import load_raw
from churn.features.builder import INPUT_COLUMNS
from churn.training.evaluate import evaluate
from churn.training.pipeline import build_pipeline

CSV_PATH = str(Path(__file__).resolve().parents[1] / "Customer-Churn-Records.csv")


def _cfg(tmp_path) -> Settings:
    return Settings(
        data_path=CSV_PATH,
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        mlflow_experiment="churn-test",
        model_name="churn-model-test",
        model_alias="production",
    )


def _fit(cfg: Settings):
    df = load_raw(cfg.data_path)
    X, y = df[INPUT_COLUMNS], df["turnover"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    pipeline = build_pipeline(random_state=cfg.random_state, n_age_bins=cfg.n_age_bins)
    pipeline.fit(X_train, y_train)
    metrics = evaluate(pipeline, X_test, y_test)
    return pipeline, X_train, metrics


def test_log_and_register_promotes_when_asked(tmp_path):
    from churn.training.registry import log_and_register

    cfg = _cfg(tmp_path)
    pipeline, X_train, metrics = _fit(cfg)
    result = log_and_register(pipeline, X_train, metrics, cfg, promote=True)

    assert result["version"] == "1"
    assert result["promoted"] is True
    client = MlflowClient(tracking_uri=cfg.mlflow_tracking_uri)
    mv = client.get_model_version_by_alias(cfg.model_name, cfg.model_alias)
    assert str(mv.version) == "1"


def test_log_and_register_skips_alias_when_not_promoting(tmp_path):
    from churn.training.registry import log_and_register

    cfg = _cfg(tmp_path)
    pipeline, X_train, metrics = _fit(cfg)
    result = log_and_register(pipeline, X_train, metrics, cfg, promote=False)

    assert result["version"] == "1"
    assert result["promoted"] is False
    client = MlflowClient(tracking_uri=cfg.mlflow_tracking_uri)
    # a version exists, but no production alias was set
    assert client.get_model_version(cfg.model_name, "1") is not None
    try:
        client.get_model_version_by_alias(cfg.model_name, cfg.model_alias)
        raise AssertionError("alias should not exist when promote=False")
    except Exception:
        pass
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn.training.registry'`

- [ ] **Step 4: Implement `log_and_register` (extracted verbatim from train.py)**

Create `src/churn/training/registry.py`:

```python
"""Shared MLflow logging + registration for the churn model.

DRY helper used by both the training entrypoint (`train.py`) and the KFP
`register_model` step, so the two never drift in how they log/register.
"""

import warnings

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

from churn.config import Settings, settings
from churn.features.builder import INPUT_COLUMNS


def log_and_register(
    pipeline, X_sample, metrics: dict, cfg: Settings = settings, promote: bool = True
) -> dict:
    """Log the fitted pipeline + params/metrics to MLflow, register a model version,
    and (optionally) move the production alias.

    `X_sample` is a DataFrame with INPUT_COLUMNS used for the signature and input example
    (a few rows suffice — the signature is schema-based). Returns run_id, version, promoted.
    """
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "model": "LogisticRegression",
                "max_iter": 500,
                "class_weight": "balanced",
                "random_state": cfg.random_state,
                "test_size": cfg.test_size,
                "n_age_bins": cfg.n_age_bins,
                "n_input_features": len(INPUT_COLUMNS),
            }
        )
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "confusion_matrix"})
        signature = infer_signature(X_sample, pipeline.predict(X_sample))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # mute pip-version + int-column hints
            info = mlflow.sklearn.log_model(
                pipeline,
                name="model",
                signature=signature,
                input_example=X_sample.head(3),
                registered_model_name=cfg.model_name,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
        version = str(info.registered_model_version)
        if promote:
            MlflowClient(tracking_uri=cfg.mlflow_tracking_uri).set_registered_model_alias(
                cfg.model_name, cfg.model_alias, version
            )
        run_id = run.info.run_id

    return {"run_id": run_id, "version": version, "promoted": promote}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_registry.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Rewire `train.py` to use the helper**

Replace the body of `train()` in `src/churn/training/train.py`. The new file (imports through the end of `train()`):

```python
"""Training entrypoint: fit the pipeline, evaluate honestly, track and register in MLflow."""

from sklearn.model_selection import train_test_split

from churn.config import Settings, settings
from churn.data import load_raw
from churn.features.builder import INPUT_COLUMNS
from churn.training.evaluate import evaluate
from churn.training.pipeline import build_pipeline
from churn.training.registry import log_and_register


def train(cfg: Settings = settings) -> dict:
    """Fit, evaluate and register the churn model. Returns run id, version and metrics."""
    df = load_raw(cfg.data_path)
    X, y = df[INPUT_COLUMNS], df["turnover"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    pipeline = build_pipeline(random_state=cfg.random_state, n_age_bins=cfg.n_age_bins)
    pipeline.fit(X_train, y_train)
    metrics = evaluate(pipeline, X_test, y_test)

    result = log_and_register(pipeline, X_train, metrics, cfg, promote=True)
    return {
        "run_id": result["run_id"],
        "version": result["version"],
        "metrics": metrics,
    }
```

Keep the existing `main()` and `if __name__ == "__main__":` block below unchanged (lines 67-81 of the original). Remove the now-unused imports (`warnings`, `mlflow`, `MlflowClient`, `infer_signature`) — they moved into `registry.py`.

- [ ] **Step 7: Run the full training suite to prove no regression**

Run: `uv run pytest tests/test_train.py tests/test_registry.py -v`
Expected: PASS (all training + registry tests green; `test_train_logs_run_and_registers_model` still passes)

- [ ] **Step 8: Lint**

Run: `uv run ruff check src tests`
Expected: no errors (if unused-import warnings appear in `train.py`, remove those imports)

- [ ] **Step 9: Commit**

```bash
git add src/churn/config.py src/churn/training/registry.py src/churn/training/train.py tests/test_registry.py
git commit -m "refactor: extract log_and_register helper and add min_roc_auc gate config"
```

---

## Task 2: `prepare_data` step

**Files:**
- Create: `src/churn/orchestration/__init__.py`
- Create: `src/churn/orchestration/steps/__init__.py`
- Create: `src/churn/orchestration/steps/prepare_data.py`
- Test: `tests/test_orchestration.py` (create)

- [ ] **Step 1: Create the package markers**

Create `src/churn/orchestration/__init__.py` (empty file, just a package marker):

```python
```

Create `src/churn/orchestration/steps/__init__.py` (empty file):

```python
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_orchestration.py`:

```python
from pathlib import Path

import pandas as pd

from churn.features.builder import INPUT_COLUMNS

CSV_PATH = str(Path(__file__).resolve().parents[1] / "Customer-Churn-Records.csv")


def test_prepare_data_writes_model_input_parquet(tmp_path):
    from churn.orchestration.steps.prepare_data import prepare_data

    out = tmp_path / "prepared.parquet"
    prepare_data(str(out), CSV_PATH)

    df = pd.read_parquet(out)
    assert list(df.columns) == INPUT_COLUMNS + ["turnover"]
    assert len(df) == 10000
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_orchestration.py::test_prepare_data_writes_model_input_parquet -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn.orchestration.steps.prepare_data'`

- [ ] **Step 4: Implement `prepare_data`**

Create `src/churn/orchestration/steps/prepare_data.py`:

```python
"""KFP step: load + validate the raw CSV and persist the model-input frame as parquet."""

from churn.data import load_raw
from churn.features.builder import INPUT_COLUMNS


def prepare_data(out_path: str, data_path: str) -> None:
    """Write INPUT_COLUMNS + turnover from the validated raw frame to a parquet at out_path."""
    df = load_raw(data_path)
    df[INPUT_COLUMNS + ["turnover"]].to_parquet(out_path, index=False)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_orchestration.py::test_prepare_data_writes_model_input_parquet -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/churn/orchestration/__init__.py src/churn/orchestration/steps/__init__.py src/churn/orchestration/steps/prepare_data.py tests/test_orchestration.py
git commit -m "feat: add prepare_data orchestration step"
```

---

## Task 3: `split_data` step

**Files:**
- Create: `src/churn/orchestration/steps/split_data.py`
- Test: `tests/test_orchestration.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_orchestration.py`:

```python
def test_split_data_reproduces_stratified_split(tmp_path):
    from churn.orchestration.steps.prepare_data import prepare_data
    from churn.orchestration.steps.split_data import split_data

    prepared = tmp_path / "prepared.parquet"
    prepare_data(str(prepared), CSV_PATH)

    train_out = tmp_path / "train.parquet"
    test_out = tmp_path / "test.parquet"
    split_data(str(prepared), str(train_out), str(test_out), test_size=0.2, random_state=42)

    train_df = pd.read_parquet(train_out)
    test_df = pd.read_parquet(test_out)

    # columns preserved on both sides
    assert list(train_df.columns) == INPUT_COLUMNS + ["turnover"]
    assert list(test_df.columns) == INPUT_COLUMNS + ["turnover"]
    # 80/20 split of 10000 rows, no rows lost
    assert len(train_df) == 8000
    assert len(test_df) == 2000
    # stratified: class balance preserved within ~1 percentage point
    assert abs(train_df["turnover"].mean() - test_df["turnover"].mean()) < 0.01
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_orchestration.py::test_split_data_reproduces_stratified_split -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn.orchestration.steps.split_data'`

- [ ] **Step 3: Implement `split_data`**

Create `src/churn/orchestration/steps/split_data.py`:

```python
"""KFP step: reproduce the train.py stratified train/test split on the prepared parquet."""

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.features.builder import INPUT_COLUMNS


def split_data(
    in_path: str, train_out: str, test_out: str, test_size: float, random_state: int
) -> None:
    """Split the prepared frame into stratified train/test parquets (same split as train.py)."""
    df = pd.read_parquet(in_path)
    X, y = df[INPUT_COLUMNS], df["turnover"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train_df = X_train.copy()
    train_df["turnover"] = y_train.to_numpy()
    test_df = X_test.copy()
    test_df["turnover"] = y_test.to_numpy()

    train_df.to_parquet(train_out, index=False)
    test_df.to_parquet(test_out, index=False)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_orchestration.py::test_split_data_reproduces_stratified_split -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/churn/orchestration/steps/split_data.py tests/test_orchestration.py
git commit -m "feat: add split_data orchestration step"
```

---

## Task 4: `train_model` step

**Files:**
- Create: `src/churn/orchestration/steps/train_model.py`
- Test: `tests/test_orchestration.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_orchestration.py`:

```python
def test_train_model_persists_fitted_pipeline(tmp_path):
    import joblib

    from churn.orchestration.steps.prepare_data import prepare_data
    from churn.orchestration.steps.split_data import split_data
    from churn.orchestration.steps.train_model import train_model

    prepared = tmp_path / "prepared.parquet"
    prepare_data(str(prepared), CSV_PATH)
    train_out = tmp_path / "train.parquet"
    test_out = tmp_path / "test.parquet"
    split_data(str(prepared), str(train_out), str(test_out), test_size=0.2, random_state=42)

    model_out = tmp_path / "model.joblib"
    train_model(str(train_out), str(model_out), random_state=42, n_age_bins=5)

    model = joblib.load(model_out)
    sample = pd.read_parquet(test_out)[INPUT_COLUMNS].head(5)
    assert len(model.predict(sample)) == 5
    assert model.predict_proba(sample).shape == (5, 2)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_orchestration.py::test_train_model_persists_fitted_pipeline -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn.orchestration.steps.train_model'`

- [ ] **Step 3: Implement `train_model`**

Create `src/churn/orchestration/steps/train_model.py`:

```python
"""KFP step: fit the churn pipeline on the train parquet and persist it with joblib."""

import joblib
import pandas as pd

from churn.features.builder import INPUT_COLUMNS
from churn.training.pipeline import build_pipeline


def train_model(train_path: str, model_out: str, random_state: int, n_age_bins: int) -> None:
    """Fit build_pipeline() on the train parquet and joblib.dump the fitted Pipeline to model_out."""
    df = pd.read_parquet(train_path)
    X, y = df[INPUT_COLUMNS], df["turnover"]
    pipeline = build_pipeline(random_state=random_state, n_age_bins=n_age_bins)
    pipeline.fit(X, y)
    joblib.dump(pipeline, model_out)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_orchestration.py::test_train_model_persists_fitted_pipeline -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/churn/orchestration/steps/train_model.py tests/test_orchestration.py
git commit -m "feat: add train_model orchestration step"
```

---

## Task 5: `evaluate_model` step

**Files:**
- Create: `src/churn/orchestration/steps/evaluate_model.py`
- Test: `tests/test_orchestration.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_orchestration.py`:

```python
def test_evaluate_model_writes_metrics_json(tmp_path):
    import json

    from churn.orchestration.steps.evaluate_model import evaluate_model
    from churn.orchestration.steps.prepare_data import prepare_data
    from churn.orchestration.steps.split_data import split_data
    from churn.orchestration.steps.train_model import train_model

    prepared = tmp_path / "prepared.parquet"
    prepare_data(str(prepared), CSV_PATH)
    train_out = tmp_path / "train.parquet"
    test_out = tmp_path / "test.parquet"
    split_data(str(prepared), str(train_out), str(test_out), test_size=0.2, random_state=42)
    model_out = tmp_path / "model.joblib"
    train_model(str(train_out), str(model_out), random_state=42, n_age_bins=5)

    metrics_out = tmp_path / "metrics.json"
    returned = evaluate_model(str(model_out), str(test_out), str(metrics_out))

    on_disk = json.loads(metrics_out.read_text())
    assert on_disk == returned
    for key in ("roc_auc", "precision", "recall", "f1", "accuracy", "confusion_matrix"):
        assert key in on_disk
    assert 0.0 < on_disk["roc_auc"] < 1.0
    assert on_disk["roc_auc"] > 0.7
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_orchestration.py::test_evaluate_model_writes_metrics_json -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn.orchestration.steps.evaluate_model'`

- [ ] **Step 3: Implement `evaluate_model`**

Create `src/churn/orchestration/steps/evaluate_model.py`:

```python
"""KFP step: evaluate a persisted model on the test parquet and write metrics JSON."""

import json

import joblib
import pandas as pd

from churn.features.builder import INPUT_COLUMNS
from churn.training.evaluate import evaluate


def evaluate_model(model_path: str, test_path: str, metrics_out: str) -> dict:
    """Load the model, evaluate on the test parquet, write metrics as JSON, and return them."""
    pipeline = joblib.load(model_path)
    df = pd.read_parquet(test_path)
    X, y = df[INPUT_COLUMNS], df["turnover"]
    metrics = evaluate(pipeline, X, y)
    with open(metrics_out, "w") as f:
        json.dump(metrics, f)
    return metrics
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_orchestration.py::test_evaluate_model_writes_metrics_json -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/churn/orchestration/steps/evaluate_model.py tests/test_orchestration.py
git commit -m "feat: add evaluate_model orchestration step"
```

---

## Task 6: `register_model` step (with quality gate)

**Files:**
- Create: `src/churn/orchestration/steps/register_model.py`
- Test: `tests/test_orchestration.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_orchestration.py`. This adds a helper that builds the four upstream artifacts once, then tests both gate branches:

```python
def _build_artifacts(tmp_path):
    """Run prepare -> split -> train -> evaluate, returning the artifact paths."""
    from churn.orchestration.steps.evaluate_model import evaluate_model
    from churn.orchestration.steps.prepare_data import prepare_data
    from churn.orchestration.steps.split_data import split_data
    from churn.orchestration.steps.train_model import train_model

    prepared = tmp_path / "prepared.parquet"
    prepare_data(str(prepared), CSV_PATH)
    train_out = tmp_path / "train.parquet"
    test_out = tmp_path / "test.parquet"
    split_data(str(prepared), str(train_out), str(test_out), test_size=0.2, random_state=42)
    model_out = tmp_path / "model.joblib"
    train_model(str(train_out), str(model_out), random_state=42, n_age_bins=5)
    metrics_out = tmp_path / "metrics.json"
    evaluate_model(str(model_out), str(test_out), str(metrics_out))
    return str(model_out), str(metrics_out), str(train_out)


def _reg_cfg(tmp_path, min_roc_auc):
    from churn.config import Settings

    return Settings(
        data_path=CSV_PATH,
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        mlflow_experiment="churn-test",
        model_name="churn-model-test",
        model_alias="production",
        min_roc_auc=min_roc_auc,
    )


def test_register_model_promotes_when_gate_passes(tmp_path):
    from mlflow import MlflowClient

    from churn.orchestration.steps.register_model import register_model

    model_path, metrics_path, train_path = _build_artifacts(tmp_path)
    cfg = _reg_cfg(tmp_path, min_roc_auc=0.5)
    result = register_model(model_path, metrics_path, train_path, cfg)

    assert result["promoted"] is True
    assert result["version"] == "1"
    assert result["roc_auc"] > 0.7
    client = MlflowClient(tracking_uri=cfg.mlflow_tracking_uri)
    mv = client.get_model_version_by_alias(cfg.model_name, cfg.model_alias)
    assert str(mv.version) == "1"


def test_register_model_skips_promotion_when_gate_fails(tmp_path):
    from mlflow import MlflowClient

    from churn.orchestration.steps.register_model import register_model

    model_path, metrics_path, train_path = _build_artifacts(tmp_path)
    cfg = _reg_cfg(tmp_path, min_roc_auc=0.99)
    result = register_model(model_path, metrics_path, train_path, cfg)

    assert result["promoted"] is False
    assert result["version"] == "1"
    client = MlflowClient(tracking_uri=cfg.mlflow_tracking_uri)
    # version registered, but no production alias set
    assert client.get_model_version(cfg.model_name, "1") is not None
    try:
        client.get_model_version_by_alias(cfg.model_name, cfg.model_alias)
        raise AssertionError("alias should not exist when gate fails")
    except Exception:
        pass
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_orchestration.py::test_register_model_promotes_when_gate_passes -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn.orchestration.steps.register_model'`

- [ ] **Step 3: Implement `register_model`**

Create `src/churn/orchestration/steps/register_model.py`:

```python
"""KFP step: register the model in MLflow, promoting to @production only if the gate passes."""

import json

import joblib
import pandas as pd

from churn.config import Settings, settings
from churn.features.builder import INPUT_COLUMNS
from churn.training.registry import log_and_register


def register_model(
    model_path: str, metrics_path: str, train_path: str, cfg: Settings = settings
) -> dict:
    """Log + register the model; move the @production alias only if roc_auc >= cfg.min_roc_auc.

    Returns run_id, version, promoted, roc_auc.
    """
    pipeline = joblib.load(model_path)
    with open(metrics_path) as f:
        metrics = json.load(f)
    X_sample = pd.read_parquet(train_path)[INPUT_COLUMNS].head(5)

    roc_auc = metrics["roc_auc"]
    promote = roc_auc >= cfg.min_roc_auc
    result = log_and_register(pipeline, X_sample, metrics, cfg, promote=promote)
    return {**result, "roc_auc": roc_auc}
```

- [ ] **Step 4: Run both gate tests to verify they pass**

Run: `uv run pytest tests/test_orchestration.py::test_register_model_promotes_when_gate_passes tests/test_orchestration.py::test_register_model_skips_promotion_when_gate_fails -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/churn/orchestration/steps/register_model.py tests/test_orchestration.py
git commit -m "feat: add register_model orchestration step with roc_auc promotion gate"
```

---

## Task 7: KFP DAG wiring + e2e test + Makefile target

Fan-in: assemble the five pure functions into `@dsl.component` wrappers, wire the `@dsl.pipeline`, add `run_local`/`main`, prove the whole DAG runs via `kfp local`, and expose `make pipeline`.

**Files:**
- Create: `src/churn/orchestration/dag.py`
- Modify: `Makefile`
- Test: `tests/test_orchestration.py` (append)

- [ ] **Step 1: Implement the DAG**

Create `src/churn/orchestration/dag.py`. Note: imports inside each `@dsl.component` are intentional (KFP executes the component function standalone), and `register_model_op` rebuilds a `Settings` from primitive params passed by the pipeline (KFP components only accept primitives/artifacts, not objects):

```python
"""KFP local pipeline: orchestrate the churn training DAG (Vertex AI Pipelines equivalent).

Run end to end with `python -m churn.orchestration.dag` (make pipeline). Uses the local
SubprocessRunner so each component runs in the current venv — no cluster, no container images.
"""

from kfp import dsl, local
from kfp.dsl import Dataset, Input, Metrics, Model, Output

from churn.config import Settings, settings


@dsl.component
def prepare_data_op(output: Output[Dataset], data_path: str):
    from churn.orchestration.steps.prepare_data import prepare_data

    prepare_data(output.path, data_path)


@dsl.component
def split_data_op(
    dataset: Input[Dataset],
    train: Output[Dataset],
    test: Output[Dataset],
    test_size: float,
    random_state: int,
):
    from churn.orchestration.steps.split_data import split_data

    split_data(dataset.path, train.path, test.path, test_size, random_state)


@dsl.component
def train_model_op(
    train_set: Input[Dataset], model: Output[Model], random_state: int, n_age_bins: int
):
    from churn.orchestration.steps.train_model import train_model

    train_model(train_set.path, model.path, random_state, n_age_bins)


@dsl.component
def evaluate_model_op(model: Input[Model], test_set: Input[Dataset], metrics: Output[Metrics]):
    from churn.orchestration.steps.evaluate_model import evaluate_model

    result = evaluate_model(model.path, test_set.path, metrics.path)
    for key in ("roc_auc", "precision", "recall", "f1", "accuracy"):
        metrics.log_metric(key, float(result[key]))


@dsl.component
def register_model_op(
    model: Input[Model],
    metrics: Input[Metrics],
    train_set: Input[Dataset],
    mlflow_tracking_uri: str,
    mlflow_experiment: str,
    model_name: str,
    model_alias: str,
    random_state: int,
    test_size: float,
    n_age_bins: int,
    min_roc_auc: float,
):
    from churn.config import Settings
    from churn.orchestration.steps.register_model import register_model

    cfg = Settings(
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
        model_name=model_name,
        model_alias=model_alias,
        random_state=random_state,
        test_size=test_size,
        n_age_bins=n_age_bins,
        min_roc_auc=min_roc_auc,
    )
    register_model(model.path, metrics.path, train_set.path, cfg)


@dsl.pipeline(name="churn-training-pipeline")
def churn_training_pipeline(
    data_path: str,
    test_size: float,
    random_state: int,
    n_age_bins: int,
    mlflow_tracking_uri: str,
    mlflow_experiment: str,
    model_name: str,
    model_alias: str,
    min_roc_auc: float,
):
    prep = prepare_data_op(data_path=data_path)
    split = split_data_op(
        dataset=prep.outputs["output"], test_size=test_size, random_state=random_state
    )
    trained = train_model_op(
        train_set=split.outputs["train"], random_state=random_state, n_age_bins=n_age_bins
    )
    ev = evaluate_model_op(model=trained.outputs["model"], test_set=split.outputs["test"])
    register_model_op(
        model=trained.outputs["model"],
        metrics=ev.outputs["metrics"],
        train_set=split.outputs["train"],
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
        model_name=model_name,
        model_alias=model_alias,
        random_state=random_state,
        test_size=test_size,
        n_age_bins=n_age_bins,
        min_roc_auc=min_roc_auc,
    )


def run_local(cfg: Settings = settings):
    """Initialize the local SubprocessRunner and execute the DAG end to end."""
    local.init(
        runner=local.SubprocessRunner(use_venv=False),
        raise_on_error=True,
        enable_caching=False,
    )
    return churn_training_pipeline(
        data_path=cfg.data_path,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        n_age_bins=cfg.n_age_bins,
        mlflow_tracking_uri=cfg.mlflow_tracking_uri,
        mlflow_experiment=cfg.mlflow_experiment,
        model_name=cfg.model_name,
        model_alias=cfg.model_alias,
        min_roc_auc=cfg.min_roc_auc,
    )


def main() -> None:
    run_local()
    print("Pipeline finished. Check MLflow for the registered model version.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the failing e2e test**

Append to `tests/test_orchestration.py`:

```python
def test_pipeline_end_to_end(tmp_path):
    from mlflow import MlflowClient

    from churn.config import Settings
    from churn.orchestration.dag import run_local

    cfg = Settings(
        data_path=CSV_PATH,
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        mlflow_experiment="churn-e2e",
        model_name="churn-model-e2e",
        model_alias="production",
        min_roc_auc=0.5,
    )

    run = run_local(cfg)
    assert run.state.name == "FINAL"

    # the DAG registered a version and promoted it (gate passes at 0.5)
    client = MlflowClient(tracking_uri=cfg.mlflow_tracking_uri)
    mv = client.get_model_version_by_alias(cfg.model_name, cfg.model_alias)
    assert str(mv.version) == "1"
```

- [ ] **Step 3: Run the e2e test to verify it fails first, then passes**

Run: `uv run pytest tests/test_orchestration.py::test_pipeline_end_to_end -v`
Expected first (before dag.py existed it would be an ImportError; since dag.py is now written in Step 1, this run should PASS). If it FAILS, read the KFP error output — the most likely cause is a component raising (paths/imports); fix and re-run until PASS.

Note: this test shells out subprocesses via the SubprocessRunner and trains the real model, so it takes ~30-60s. That is expected.

- [ ] **Step 4: Add the Makefile target**

In `Makefile`, add `pipeline` to the `.PHONY` line (line 3) and add the target after the `train` target (after line 24):

Update line 3 to include `pipeline`:

```makefile
.PHONY: help setup test lint format train pipeline serve feast-materialize docker-build docker-run
```

Add the target after the `train` block:

```makefile
pipeline: ## Roda o DAG de treino KFP local ponta a ponta (prepare -> split -> train -> evaluate -> register)
	uv run python -m churn.orchestration.dag
```

- [ ] **Step 5: Run the full suite + lint**

Run: `uv run pytest -v`
Expected: all tests green (53 prior + registry + orchestration).

Run: `uv run ruff check src tests`
Expected: no errors.

- [ ] **Step 6: Smoke-test `make pipeline` end to end**

Run: `make pipeline`
Expected: KFP logs each task (`prepare-data-op` … `register-model-op`) finishing with SUCCESS, ending with `Pipeline ... finished with status SUCCESS` and the printed line `Pipeline finished. Check MLflow for the registered model version.` This writes to the real `mlflow.db`/`mlruns/` (both gitignored).

- [ ] **Step 7: Commit**

```bash
git add src/churn/orchestration/dag.py tests/test_orchestration.py Makefile
git commit -m "feat: wire KFP local churn training DAG and add make pipeline target"
```

---

## Self-Review

**1. Spec coverage:**
- 5-stage DAG with artifact lineage (§1.1, §3.5) → Tasks 2-6 (pure steps) + Task 7 (wiring). ✓
- SubprocessRunner + doc mapping (§2, §8) → Task 7 `run_local`; mapping table lives in the spec. ✓
- Pure step / thin wrapper separation (§3) → Tasks 2-6 are pure functions; Task 7 wrappers delegate. ✓
- Artifacts Dataset/Model/Metrics (§3.2) → parquet Datasets (Tasks 2-3), joblib Model (Task 4), JSON+dsl.Metrics (Task 5, Task 7 `log_metric`). ✓
- DRY `log_and_register` (§3.3) → Task 1. ✓
- `min_roc_auc` gate (§3.4) → Task 1 (config) + Task 6 (gate logic, both branches tested). ✓
- Config as pipeline parameters, not env (§3.5) → Task 7 pipeline params + `register_model_op` rebuilds Settings. ✓
- Makefile `pipeline` (§3.6) → Task 7 Step 4. ✓
- `.gitignore local_outputs/` (§3.7) → already committed (`593b4f7`); no task needed. ✓
- Testing: 5 per-step unit tests + e2e (§6) → Tasks 2-6 unit tests + Task 7 e2e; regression via Task 1 Step 7. ✓
- train.py behavior preserved (§1) → Task 1 Step 7 runs `tests/test_train.py`. ✓

**2. Placeholder scan:** No TBD/TODO. Every code step shows complete code; every run step shows the exact command and expected result. The only judgment call ("if it FAILS, read the KFP error") is in Task 7 Step 3, which is legitimate debugging guidance for the one integration test, not a placeholder for missing code.

**3. Type consistency:**
- `log_and_register(pipeline, X_sample, metrics, cfg, promote)` → same signature in Task 1 (def), Task 1 train.py (call), Task 6 register_model (call). ✓
- Return dict keys `{run_id, version, promoted}` from `log_and_register`; `register_model` adds `roc_auc`. Tests assert exactly these. ✓
- Step function signatures match between their `def` (Tasks 2-6) and the wrapper calls in `dag.py` (Task 7): `prepare_data(out_path, data_path)`, `split_data(in_path, train_out, test_out, test_size, random_state)`, `train_model(train_path, model_out, random_state, n_age_bins)`, `evaluate_model(model_path, test_path, metrics_out)`, `register_model(model_path, metrics_path, train_path, cfg)`. ✓
- KFP output keys used in wiring (`prep.outputs["output"]`, `split.outputs["train"]/["test"]`, `trained.outputs["model"]`, `ev.outputs["metrics"]`) match the `Output[...]` param names in the wrappers. ✓

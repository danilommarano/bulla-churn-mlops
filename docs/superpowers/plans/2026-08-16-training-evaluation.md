# Milestone 2 — Training & Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the two loose churn scripts into a single, persisted `sklearn.Pipeline` with honest evaluation metrics, tracked and versioned in a local MLflow, consuming the leakage-free `GeographyChurnRateEncoder` from Milestone 1.

**Architecture:** One `sklearn.Pipeline` = `ChurnFeatureBuilder` (learns `age_bucket` quantile edges and `geography_churn_rate` on TRAIN only; derives `balance_per_product`) → `ColumnTransformer` (OneHot categoricals + StandardScaler numericals + passthrough) → `LogisticRegression(class_weight="balanced")`. Because every fitted transform lives inside the persisted pipeline, nothing is re-fitted at inference (kills train/serve skew). Training logs params/metrics/model to a local MLflow (SQLite backend), registers the model and points a `@production` alias at the new version.

**Tech Stack:** Python 3.12, uv, scikit-learn 1.9, MLflow 3.15 (local SQLite backend), pandas 2.x, pytest, ruff.

**Bugs closed (design §2/§6):** #2 (skew — everything inside one persisted Pipeline), #3 (`CustomerId` dropped), #4 (`surname_encoded` dropped), #5 (honest AUC/precision/recall/F1 instead of accuracy), #6 (`stratify`+`random_state` split), #8 (MLflow Registry with signature + alias instead of a loose pickle).

**Verified before writing:** the full pipeline was prototyped on the real 10k-row CSV — `roc_auc=0.7644`, joblib round-trip identical, reproducible across two fits; the MLflow SQLite register→alias→load flow works end to end.

---

## File Structure

- `src/churn/config.py` — **modify**: add MLflow + model-registry settings and `n_age_bins`.
- `src/churn/features/builder.py` — **create**: `ChurnFeatureBuilder` + input-column constants.
- `src/churn/training/__init__.py` — **create**: empty package marker.
- `src/churn/training/pipeline.py` — **create**: `build_pipeline()` + column-group constants.
- `src/churn/training/evaluate.py` — **create**: `evaluate()` honest metrics.
- `src/churn/training/train.py` — **create**: `train()` orchestration + MLflow logging/registry + `main()`.
- `tests/test_config.py` — **modify**: assert new defaults.
- `tests/test_builder.py` — **create**.
- `tests/test_pipeline.py` — **create**.
- `tests/test_evaluate.py` — **create**.
- `tests/test_train.py` — **create**: MLflow smoke/integration test.
- `Makefile` — **modify**: add `train` target.
- `.env.example` — **modify**: document new `CHURN_` vars.

`.gitignore` already ignores `mlruns/`, `mlartifacts/`, `*.db` — no change needed.

---

## Task 0: Dependency compatibility — ALREADY DONE ✅

**Context (do not redo):** MLflow ≥3 (and the wider MLOps stack) requires `pandas<3`, but Milestone 1 pinned `pandas>=3.0.5`, making the milestone unsatisfiable. Committed as `build: add mlflow and relax pandas pin to <3 for MLOps-stack compatibility` (commit `5155594`): `pyproject.toml` now has `mlflow>=3.15.1` and `pandas>=2.2,<3` (numpy stays `>=2.5.2`). All 17 Milestone 1 tests pass under pandas 2.3.3; ruff clean. **Start implementation at Task 1.**

---

## Task 1: Config — MLflow & registry settings

**Files:**
- Modify: `src/churn/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_mlflow_and_model_defaults():
    s = Settings()
    assert s.n_age_bins == 5
    assert s.mlflow_tracking_uri == "sqlite:///mlflow.db"
    assert s.mlflow_experiment == "churn"
    assert s.model_name == "churn-model"
    assert s.model_alias == "production"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_mlflow_and_model_defaults -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'n_age_bins'`.

- [ ] **Step 3: Add the fields**

Replace the field block in `src/churn/config.py` (the three lines `data_path`/`random_state`/`test_size`) with:

```python
    data_path: str = "Customer-Churn-Records.csv"
    random_state: int = 42
    test_size: float = 0.2
    n_age_bins: int = 5

    # MLflow (local, no server) + model registry
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "churn"
    model_name: str = "churn-model"
    model_alias: str = "production"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (all 3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/churn/config.py tests/test_config.py
git commit -m "feat: add MLflow and model-registry settings"
```

---

## Task 2: `ChurnFeatureBuilder` — leakage-free feature engineering

**Files:**
- Create: `src/churn/features/builder.py`
- Test: `tests/test_builder.py`

**What it does:** a scikit-learn transformer. `fit(X, y)` learns the `Geography`→churn-rate mapping (via the Milestone-1 `GeographyChurnRateEncoder`, using `y` only) and the `Age` quantile bin edges (via `KBinsDiscretizer`). `transform(X)` returns a DataFrame with three derived columns added: `balance_per_product` (deterministic), `geography_churn_rate`, `age_bucket`. It fills `Balance`/`EstimatedSalary` nulls with 0 (they are nullable in the schema). `CustomerId` and `surname_encoded` are simply never selected (bugs #3, #4).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_builder.py`:

```python
import numpy as np
import pandas as pd
import pytest

from churn.features.builder import ChurnFeatureBuilder, INPUT_COLUMNS


def _raw(n_geo, geos, ages, balances, products, target):
    """Minimal frame carrying every INPUT column the builder reads."""
    base = {
        "CreditScore": [650] * n_geo,
        "Age": ages,
        "Balance": balances,
        "EstimatedSalary": [50000.0] * n_geo,
        "Tenure": [5] * n_geo,
        "NumOfProducts": products,
        "HasCrCard": [1] * n_geo,
        "Satisfaction Score": [3] * n_geo,
        "Point Earned": [500] * n_geo,
        "Geography": geos,
        "Gender": ["Male"] * n_geo,
        "Card Type": ["GOLD"] * n_geo,
    }
    return pd.DataFrame(base), pd.Series(target)


def test_transform_adds_derived_columns():
    X, y = _raw(4, ["Sao Paulo"] * 4, [30, 40, 50, 60], [1000.0, 2000.0, 0.0, 4000.0],
                [1, 2, 1, 4], [1, 0, 0, 1])
    out = ChurnFeatureBuilder().fit(X, y).transform(X)
    for col in ("balance_per_product", "geography_churn_rate", "age_bucket"):
        assert col in out.columns


def test_balance_per_product_and_null_fill():
    X, y = _raw(2, ["Sao Paulo", "Sao Paulo"], [30, 40], [np.nan, 2000.0], [2, 4], [1, 0])
    out = ChurnFeatureBuilder().fit(X, y).transform(X)
    # null Balance -> 0 -> 0/2 = 0.0 ; 2000/4 = 500.0
    assert list(out["balance_per_product"]) == [0.0, 500.0]


def test_geography_rate_is_learned_on_train_only():
    # train: SP churn rate 0.5 ; test carries different labels that must NOT leak
    Xtr, ytr = _raw(2, ["Sao Paulo", "Sao Paulo"], [30, 40], [100.0, 200.0], [1, 1], [1, 0])
    Xte, _ = _raw(2, ["Sao Paulo", "Sao Paulo"], [30, 40], [100.0, 200.0], [1, 1], [1, 1])
    out = ChurnFeatureBuilder().fit(Xtr, ytr).transform(Xte)
    assert list(out["geography_churn_rate"]) == [0.5, 0.5]


def test_age_bucket_in_range():
    ages = list(range(20, 40))
    X, y = _raw(20, ["Sao Paulo"] * 20, ages, [1000.0] * 20, [1] * 20, [i % 2 for i in range(20)])
    out = ChurnFeatureBuilder(n_age_bins=5).fit(X, y).transform(X)
    assert out["age_bucket"].min() >= 0
    assert out["age_bucket"].max() <= 4


def test_transform_before_fit_raises():
    X, _ = _raw(1, ["Sao Paulo"], [30], [1000.0], [1], [0])
    with pytest.raises(RuntimeError):
        ChurnFeatureBuilder().transform(X)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_builder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn.features.builder'`.

- [ ] **Step 3: Implement the builder**

Create `src/churn/features/builder.py`:

```python
"""Leakage-free feature engineering as a scikit-learn transformer.

Learns everything that "sees" the data (age-bucket edges, geography churn rate)
on the TRAIN split only; deterministic derivations (balance per product) need no
fit. Dropping CustomerId and surname_encoded is done by simply never selecting
them (fixes bugs #3 and #4).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer

from churn.features.aggregations import GeographyChurnRateEncoder

# Raw columns fed into the pipeline (CustomerId, Surname, RowNumber, turnover,
# Complain and IsActiveMember are intentionally excluded — see design §2).
RAW_NUMERIC = [
    "CreditScore",
    "Age",
    "Balance",
    "EstimatedSalary",
    "Tenure",
    "NumOfProducts",
    "HasCrCard",
    "Satisfaction Score",
    "Point Earned",
]
RAW_CATEGORICAL = ["Geography", "Gender", "Card Type"]
INPUT_COLUMNS = RAW_NUMERIC + RAW_CATEGORICAL


class ChurnFeatureBuilder(BaseEstimator, TransformerMixin):
    """Adds balance_per_product, geography_churn_rate and age_bucket to the frame."""

    def __init__(self, n_age_bins: int = 5, random_state: int = 42):
        self.n_age_bins = n_age_bins
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None) -> "ChurnFeatureBuilder":
        if y is None:
            raise ValueError("ChurnFeatureBuilder.fit requires y (the target).")
        geo_frame = pd.DataFrame(
            {"Geography": X["Geography"].to_numpy(), "turnover": np.asarray(y)}
        )
        self.geo_ = GeographyChurnRateEncoder().fit(geo_frame)
        self.age_ = KBinsDiscretizer(
            n_bins=self.n_age_bins,
            encode="ordinal",
            strategy="quantile",
            quantile_method="averaged_inverted_cdf",
        )
        self.age_.fit(X[["Age"]])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "geo_"):
            raise RuntimeError("Call fit() before transform().")
        X = X.copy()
        X["Balance"] = X["Balance"].fillna(0.0)
        X["EstimatedSalary"] = X["EstimatedSalary"].fillna(0.0)
        X["balance_per_product"] = X["Balance"] / X["NumOfProducts"]
        X["geography_churn_rate"] = self.geo_.transform(X[["Geography"]]).to_numpy()
        X["age_bucket"] = self.age_.transform(X[["Age"]]).ravel()
        return X
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_builder.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/churn/features/builder.py tests/test_builder.py
git commit -m "feat: add leakage-free ChurnFeatureBuilder transformer"
```

---

## Task 3: Training pipeline factory

**Files:**
- Create: `src/churn/training/__init__.py`
- Create: `src/churn/training/pipeline.py`
- Test: `tests/test_pipeline.py`

**What it does:** `build_pipeline()` assembles the full estimator. The `ColumnTransformer` scales the six continuous columns, one-hot-encodes the three categoricals (`handle_unknown="ignore"` so an unseen category at inference is all-zeros instead of an error), and passes the remaining engineered/ordinal columns straight through. `LogisticRegression(class_weight="balanced")` counters the ~20% class imbalance without changing the algorithm.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline.py`:

```python
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from churn.data import load_raw
from churn.features.builder import INPUT_COLUMNS
from churn.training.pipeline import build_pipeline

CSV_PATH = str(Path(__file__).resolve().parents[1] / "Customer-Churn-Records.csv")


def _split():
    df = load_raw(CSV_PATH)
    X, y = df[INPUT_COLUMNS], df["turnover"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def test_pipeline_fits_and_predicts():
    X_train, X_test, y_train, y_test = _split()
    pipe = build_pipeline().fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert set(np.unique(pipe.predict(X_test))) <= {0, 1}


def test_pipeline_survives_persist_roundtrip():
    """The whole fitted pipeline is self-contained: reload gives identical output (bug #2)."""
    X_train, X_test, y_train, _ = _split()
    pipe = build_pipeline().fit(X_train, y_train)
    before = pipe.predict_proba(X_test)[:, 1]
    path = Path(tempfile.mkdtemp()) / "pipe.pkl"
    joblib.dump(pipe, path)
    after = joblib.load(path).predict_proba(X_test)[:, 1]
    assert np.allclose(before, after)


def test_pipeline_is_reproducible():
    """Same seed -> same predictions (bug #6)."""
    X_train, X_test, y_train, _ = _split()
    a = build_pipeline(random_state=42).fit(X_train, y_train).predict_proba(X_test)[:, 1]
    b = build_pipeline(random_state=42).fit(X_train, y_train).predict_proba(X_test)[:, 1]
    assert np.allclose(a, b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn.training'`.

- [ ] **Step 3: Create the package marker and the factory**

Create `src/churn/training/__init__.py`:

```python
"""Training subpackage: pipeline factory, evaluation and the training entrypoint."""
```

Create `src/churn/training/pipeline.py`:

```python
"""Assembles the single, persistable churn estimator (features + preprocessing + model)."""

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.features.builder import RAW_CATEGORICAL, ChurnFeatureBuilder

# Columns standard-scaled (continuous). balance_per_product is produced by the builder.
SCALE_COLUMNS = [
    "CreditScore",
    "Age",
    "Balance",
    "EstimatedSalary",
    "Point Earned",
    "balance_per_product",
]
# Already-bounded / engineered columns that need no scaling or encoding.
PASSTHROUGH_COLUMNS = [
    "Tenure",
    "NumOfProducts",
    "HasCrCard",
    "Satisfaction Score",
    "geography_churn_rate",
    "age_bucket",
]


def build_pipeline(random_state: int = 42, n_age_bins: int = 5) -> Pipeline:
    """Return the full unfitted churn pipeline."""
    preprocess = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), SCALE_COLUMNS),
            ("onehot", OneHotEncoder(handle_unknown="ignore"), RAW_CATEGORICAL),
            ("pass", "passthrough", PASSTHROUGH_COLUMNS),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("features", ChurnFeatureBuilder(n_age_bins=n_age_bins, random_state=random_state)),
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(
                    max_iter=500, class_weight="balanced", random_state=random_state
                ),
            ),
        ]
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/churn/training/__init__.py src/churn/training/pipeline.py tests/test_pipeline.py
git commit -m "feat: add churn training pipeline factory"
```

---

## Task 4: Honest evaluation metrics

**Files:**
- Create: `src/churn/training/evaluate.py`
- Test: `tests/test_evaluate.py`

**What it does:** replaces the misleading `accuracy` (bug #5) with ROC-AUC, precision, recall, F1 and the confusion matrix — the metrics that actually matter on a ~20%-churn base.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_evaluate.py`:

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from churn.training.evaluate import evaluate


def _perfectly_separable():
    X = pd.DataFrame({"x": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]})
    y = pd.Series([0, 0, 0, 1, 1, 1])
    return LogisticRegression().fit(X, y), X, y


def test_evaluate_returns_all_metrics():
    model, X, y = _perfectly_separable()
    m = evaluate(model, X, y)
    assert set(m) == {"roc_auc", "precision", "recall", "f1", "accuracy", "confusion_matrix"}


def test_metrics_are_in_range_and_high_on_separable_data():
    model, X, y = _perfectly_separable()
    m = evaluate(model, X, y)
    for key in ("roc_auc", "precision", "recall", "f1", "accuracy"):
        assert 0.0 <= m[key] <= 1.0
    assert m["roc_auc"] == 1.0


def test_confusion_matrix_is_serializable_2x2():
    model, X, y = _perfectly_separable()
    m = evaluate(model, X, y)
    assert isinstance(m["confusion_matrix"], list)
    assert np.array(m["confusion_matrix"]).shape == (2, 2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn.training.evaluate'`.

- [ ] **Step 3: Implement evaluate**

Create `src/churn/training/evaluate.py`:

```python
"""Honest classification metrics for an imbalanced churn base (fixes bug #5)."""

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(model, X, y) -> dict:
    """Compute ROC-AUC, precision, recall, F1, accuracy and the confusion matrix.

    `confusion_matrix` is returned as a nested list so it is JSON/MLflow friendly.
    """
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/churn/training/evaluate.py tests/test_evaluate.py
git commit -m "feat: add honest evaluation metrics"
```

---

## Task 5: Training orchestration + MLflow tracking/registry

**Files:**
- Create: `src/churn/training/train.py`
- Test: `tests/test_train.py`
- Modify: `Makefile`
- Modify: `.env.example`

**What it does:** `train(cfg)` loads+validates the CSV, splits with `stratify`+`random_state`, fits the pipeline, evaluates it, then logs params/metrics/model to MLflow (local SQLite backend), registers the model under `cfg.model_name` and points the `cfg.model_alias` (`production`) alias at the new version — the modern MLflow-3 replacement for the deprecated Staging→Production stages (closes bug #8). `main()` wires it to `make train`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_train.py`:

```python
from pathlib import Path

from mlflow import MlflowClient
from mlflow.sklearn import load_model

from churn.config import Settings
from churn.features.builder import INPUT_COLUMNS
from churn.training.train import train

CSV_PATH = str(Path(__file__).resolve().parents[1] / "Customer-Churn-Records.csv")


def _isolated_settings(tmp_path) -> Settings:
    return Settings(
        data_path=CSV_PATH,
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        mlflow_experiment="churn-test",
        model_name="churn-model-test",
        model_alias="production",
    )


def test_train_logs_run_and_registers_model(tmp_path):
    cfg = _isolated_settings(tmp_path)
    result = train(cfg)

    # honest metric was computed and is plausible for this dataset
    assert result["metrics"]["roc_auc"] > 0.7
    assert result["version"] == "1"

    # the alias points at the freshly trained version, and it loads + predicts
    client = MlflowClient(tracking_uri=cfg.mlflow_tracking_uri)
    mv = client.get_model_version_by_alias(cfg.model_name, cfg.model_alias)
    assert mv.version == result["version"]

    model = load_model(f"models:/{cfg.model_name}@{cfg.model_alias}")
    preds = model.predict.__self__  # sanity: it is a loaded estimator
    assert preds is not None
    # it accepts the same input schema the pipeline was trained on
    from churn.data import load_raw

    sample = load_raw(cfg.data_path)[INPUT_COLUMNS].head(5)
    assert len(model.predict(sample)) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn.training.train'`.

- [ ] **Step 3: Implement train + main**

Create `src/churn/training/train.py`:

```python
"""Training entrypoint: fit the pipeline, evaluate honestly, track and register in MLflow."""

import warnings

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split

from churn.config import Settings, settings
from churn.data import load_raw
from churn.features.builder import INPUT_COLUMNS
from churn.training.evaluate import evaluate
from churn.training.pipeline import build_pipeline


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
        signature = infer_signature(X_train, pipeline.predict(X_train))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # mute pip-version + int-column hints
            info = mlflow.sklearn.log_model(
                pipeline,
                name="model",
                signature=signature,
                input_example=X_train.head(3),
                registered_model_name=cfg.model_name,
            )
        MlflowClient(tracking_uri=cfg.mlflow_tracking_uri).set_registered_model_alias(
            cfg.model_name, cfg.model_alias, info.registered_model_version
        )
        run_id = run.info.run_id

    return {
        "run_id": run_id,
        "version": info.registered_model_version,
        "metrics": metrics,
    }


def main() -> None:
    result = train()
    m = result["metrics"]
    print(
        f"roc_auc={m['roc_auc']:.4f} precision={m['precision']:.4f} "
        f"recall={m['recall']:.4f} f1={m['f1']:.4f} accuracy={m['accuracy']:.4f}"
    )
    print(
        f"registered '{settings.model_name}' v{result['version']} "
        f"@{settings.model_alias} (run {result['run_id']})"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_train.py -v`
Expected: PASS (1 test). It trains on the full 10k rows and exercises the real MLflow SQLite flow in a tmp dir; allow a few seconds.

- [ ] **Step 5: Add the Makefile target and document env vars**

In `Makefile`, add `train` to the `.PHONY` line:

```makefile
.PHONY: help setup test lint format train
```

And append this target after the `format` target:

```makefile
train: ## Treina o modelo e registra no MLflow (backend SQLite local)
	uv run python -m churn.training.train
```

Append to `.env.example`:

```bash
CHURN_N_AGE_BINS=5
CHURN_MLFLOW_TRACKING_URI=sqlite:///mlflow.db
CHURN_MLFLOW_EXPERIMENT=churn
CHURN_MODEL_NAME=churn-model
CHURN_MODEL_ALIAS=production
```

- [ ] **Step 6: Verify the end-to-end target runs**

Run: `make train`
Expected: prints a line like `roc_auc=0.76... precision=0.35...` and `registered 'churn-model' v1 @production ...`; creates `mlflow.db` and `mlartifacts/` (both git-ignored).

- [ ] **Step 7: Commit**

```bash
git add src/churn/training/train.py tests/test_train.py Makefile .env.example
git commit -m "feat: add MLflow-tracked training entrypoint with model registry"
```

---

## Final verification (after all tasks)

```bash
uv run pytest -q          # expect: all Milestone 1 + Milestone 2 tests green
uv run ruff check .       # expect: All checks passed!
uv run ruff format --check src tests
```

Then use **superpowers:finishing-a-development-branch** to open the PR referencing `Closes #5`.

---

## Self-Review

**Spec coverage (design phase 3 = "Treino & avaliação — sklearn Pipeline + métricas + MLflow tracking/registry"):**
- sklearn Pipeline (ColumnTransformer + LogisticRegression class_weight=balanced) → Task 3 ✅
- Consumes Milestone-1 encoder → Task 2 (`GeographyChurnRateEncoder` inside `ChurnFeatureBuilder`) ✅
- Drops `CustomerId`/`surname_encoded` → Task 2 (`INPUT_COLUMNS` excludes them) ✅
- `stratify`+`random_state` split → Task 5 ✅
- Honest metrics AUC/precision/recall/F1/confusion → Task 4 ✅
- MLflow tracking + registry with signature + stage/alias → Task 5 ✅
- No train/serve skew (single persisted pipeline) → Task 3 persist round-trip test ✅
- Bugs #2/#3/#4/#5/#6/#8 each have a closing task ✅

**Out of scope (correctly deferred):** Feast serving of the geography feature (Milestone 3), KFP DAG (Milestone 4), FastAPI + `scoring.py` wiring at inference (Milestone 5), Evidently (Milestone 6). `scoring.py` already exists and is tested; it is consumed at serving time, not training time.

**Placeholder scan:** none — every code step carries complete code.

**Type consistency:** `INPUT_COLUMNS` (builder.py) is the single source of feature columns, imported by pipeline tests, `train.py` and `test_train.py`. `SCALE_COLUMNS`/`PASSTHROUGH_COLUMNS`/`RAW_CATEGORICAL` are referenced consistently. `train()` returns `{"run_id", "version", "metrics"}`, matching every assertion in `test_train.py`. `evaluate()` returns the six keys asserted in `test_evaluate.py`.

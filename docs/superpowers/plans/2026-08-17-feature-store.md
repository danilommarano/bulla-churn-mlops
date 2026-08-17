# Feast Feature Store Implementation Plan (Marco 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve the aggregated `geography_churn_rate` feature through a local Feast offline/online store and prove online==offline==training consistency (anti-skew), without changing the existing training or serving pipeline.

**Architecture:** A new `src/churn/feature_store/` package defines a Feast `Entity`/`FeatureView` programmatically (no `feature_store.yaml`), populates a parquet offline store from the SAME train split the model uses (reusing the leakage-safe `GeographyChurnRateEncoder`), materializes it into a SQLite online store, and reads it back by entity key with a global fallback. A defensive read-only API endpoint demonstrates online serving; tests prove the served value equals the value the model learned.

**Tech Stack:** Python 3.12, Feast 0.65 (already in `pyproject.toml` from Fase A), pandas 2.x, scikit-learn, FastAPI, pytest, uv, ruff.

**Spec:** [`docs/superpowers/specs/2026-08-17-feature-store.md`](../specs/2026-08-17-feature-store.md)

---

## Domain facts (read before starting)

- Valid geographies in this dataset are **`Minas Gerais`, `Rio de Janeiro`, `Sao Paulo`** (see `src/churn/schema.py:6`). There is no France/Germany/Spain here.
- The target column is `turnover` (0/1).
- The train split used everywhere is `train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y)` with `X = df[INPUT_COLUMNS]`, `y = df["turnover"]` (see `src/churn/training/train.py:20-23`).
- `GeographyChurnRateEncoder` (`src/churn/features/aggregations.py`) exposes `.mapping_` (per-geography mean) and `.global_rate_` (fallback) after `fit(df)` where `df` has columns `Geography` and `turnover`.
- Run tests with `uv run pytest`. Lint with `uv run ruff check src tests`.

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `src/churn/config.py` | modify | Add `feast_repo_path` + derived path properties |
| `src/churn/feature_store/__init__.py` | create | Package marker |
| `src/churn/feature_store/definitions.py` | create | Entity, FileSource, FeatureView, `build_store`, constants |
| `src/churn/feature_store/materialize.py` | create | `build_offline_frame`, `materialize`, `main` |
| `src/churn/feature_store/store.py` | create | `get_geography_churn_rate`, `FeatureStoreUnavailable` |
| `src/churn/serving/api.py` | modify | `GET /features/geography/{geography}` endpoint |
| `tests/test_config.py` | modify | Add Feast path-composition test |
| `tests/test_feature_store.py` | create | Definitions, offline frame, online consistency, fallback, endpoint |
| `.gitignore` | modify | Ignore `feature_repo/` |
| `Makefile` | modify | `feast-materialize` target + `docker-run` mount |

---

## Task 1: Config — Feast repo path and derived paths

**Files:**
- Modify: `src/churn/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_feast_paths_compose():
    s = Settings(feast_repo_path="/tmp/fr")
    assert s.feast_registry_path == "/tmp/fr/registry.db"
    assert s.feast_online_path == "/tmp/fr/online_store.db"
    assert s.feast_offline_path == "/tmp/fr/data/geo_churn_stats.parquet"


def test_feast_repo_path_default():
    assert Settings().feast_repo_path == "feature_repo"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_feast_paths_compose -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'feast_repo_path'`

- [ ] **Step 3: Implement**

In `src/churn/config.py`, add the `Path` import at the top (below the existing import):

```python
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
```

Add the field after `model_alias` (inside the `Settings` class, before `settings = Settings()`):

```python
    # Feast local feature store (offline parquet + online SQLite, no cloud)
    feast_repo_path: str = "feature_repo"

    @property
    def feast_registry_path(self) -> str:
        return str(Path(self.feast_repo_path) / "registry.db")

    @property
    def feast_online_path(self) -> str:
        return str(Path(self.feast_repo_path) / "online_store.db")

    @property
    def feast_offline_path(self) -> str:
        return str(Path(self.feast_repo_path) / "data" / "geo_churn_stats.parquet")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (all config tests, including the two new ones)

- [ ] **Step 5: Commit**

```bash
git add src/churn/config.py tests/test_config.py
git commit -m "feat: add Feast repo path settings"
```

---

## Task 2: Feast definitions and store factory

**Files:**
- Create: `src/churn/feature_store/__init__.py`
- Create: `src/churn/feature_store/definitions.py`
- Test: `tests/test_feature_store.py`

- [ ] **Step 1: Create the package marker**

Create `src/churn/feature_store/__init__.py` with a single line:

```python
"""Local Feast feature store for the churn model (Vertex AI Feature Store equivalent)."""
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_feature_store.py` with the shared header and the first test:

```python
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from churn.config import Settings
from churn.data import load_raw
from churn.features.aggregations import GeographyChurnRateEncoder
from churn.features.builder import INPUT_COLUMNS
from churn.feature_store.definitions import (
    FEATURE_NAME,
    FEATURE_VIEW_NAME,
    GLOBAL_KEY,
    build_feature_view,
    geography,
)

CSV_PATH = str(Path(__file__).resolve().parents[1] / "Customer-Churn-Records.csv")


def _cfg(repo: Path) -> Settings:
    return Settings(data_path=CSV_PATH, feast_repo_path=str(repo / "feature_repo"))


def _train_encoder(cfg: Settings) -> GeographyChurnRateEncoder:
    df = load_raw(cfg.data_path)
    X, y = df[INPUT_COLUMNS], df["turnover"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    frame = pd.DataFrame(
        {"Geography": X_train["Geography"].to_numpy(), "turnover": y_train.to_numpy()}
    )
    return GeographyChurnRateEncoder().fit(frame)


def test_definitions_build(tmp_path):
    cfg = _cfg(tmp_path)
    assert geography.join_keys == ["Geography"]
    fv = build_feature_view(cfg)
    assert fv.name == FEATURE_VIEW_NAME == "geo_churn_stats"
    assert FEATURE_NAME in [f.name for f in fv.features]
    assert GLOBAL_KEY == "__global__"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_feature_store.py::test_definitions_build -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn.feature_store.definitions'`

- [ ] **Step 4: Implement `definitions.py`**

Create `src/churn/feature_store/definitions.py`:

```python
"""Feast definitions and a programmatic store factory (dispenses feature_store.yaml)."""

from datetime import timedelta

from feast import Entity, FeatureStore, FeatureView, Field, FileSource, ValueType
from feast.infra.online_stores.sqlite import SqliteOnlineStoreConfig
from feast.repo_config import RepoConfig
from feast.types import Float32

from churn.config import Settings, settings

# Sentinel entity key that stores the global fallback rate (unseen geography).
GLOBAL_KEY = "__global__"
FEATURE_VIEW_NAME = "geo_churn_stats"
FEATURE_NAME = "geography_churn_rate"

# Entity keyed by Geography (Vertex AI "Entity Type").
geography = Entity(
    name="geography",
    join_keys=["Geography"],
    value_type=ValueType.STRING,
)


def _file_source(cfg: Settings) -> FileSource:
    return FileSource(
        name="geo_churn_source",
        path=cfg.feast_offline_path,
        timestamp_field="event_timestamp",
    )


def build_feature_view(cfg: Settings = settings) -> FeatureView:
    return FeatureView(
        name=FEATURE_VIEW_NAME,
        entities=[geography],
        ttl=timedelta(days=3650),
        schema=[Field(name=FEATURE_NAME, dtype=Float32)],
        source=_file_source(cfg),
        online=True,
    )


def build_store(cfg: Settings = settings) -> FeatureStore:
    config = RepoConfig(
        project="churn",
        provider="local",
        registry=cfg.feast_registry_path,
        online_store=SqliteOnlineStoreConfig(path=cfg.feast_online_path),
        entity_key_serialization_version=3,
    )
    return FeatureStore(config=config)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_feature_store.py::test_definitions_build -v`
Expected: PASS

Note: if Feast rejects `value_type=` on `Entity` in the installed version, drop that kwarg (the join-key type is inferred as string). If `fv.features` is empty, assert against `[f.name for f in fv.schema]` instead. These are the only two API-surface adjustments this task might need.

- [ ] **Step 6: Commit**

```bash
git add src/churn/feature_store/__init__.py src/churn/feature_store/definitions.py tests/test_feature_store.py
git commit -m "feat: add Feast entity, feature view and store factory"
```

---

## Task 3: Offline frame builder + materialization

**Files:**
- Create: `src/churn/feature_store/materialize.py`
- Test: `tests/test_feature_store.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_feature_store.py`:

```python
def test_build_offline_frame_matches_encoder(tmp_path):
    from churn.feature_store.materialize import build_offline_frame

    cfg = _cfg(tmp_path)
    frame = build_offline_frame(cfg)
    encoder = _train_encoder(cfg)

    assert "event_timestamp" in frame.columns
    assert GLOBAL_KEY in set(frame["Geography"])

    lookup = dict(zip(frame["Geography"], frame[FEATURE_NAME]))
    for geo, rate in encoder.mapping_.items():
        assert lookup[geo] == pytest.approx(rate, abs=1e-9)
    assert lookup[GLOBAL_KEY] == pytest.approx(encoder.global_rate_, abs=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_feature_store.py::test_build_offline_frame_matches_encoder -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn.feature_store.materialize'`

- [ ] **Step 3: Implement `materialize.py`**

Create `src/churn/feature_store/materialize.py`:

```python
"""Populate the offline parquet from the train split and materialize into the online store."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import Settings, settings
from churn.data import load_raw
from churn.feature_store.definitions import (
    FEATURE_NAME,
    GLOBAL_KEY,
    build_feature_view,
    build_store,
    geography,
)
from churn.features.aggregations import GeographyChurnRateEncoder
from churn.features.builder import INPUT_COLUMNS

# Fixed timestamps keep materialization reproducible (no wall-clock dependence).
_EVENT_TS = pd.Timestamp("2025-01-01", tz="UTC")
_MATERIALIZE_END = datetime(2025, 1, 2, tzinfo=timezone.utc)


def build_offline_frame(cfg: Settings = settings) -> pd.DataFrame:
    """Fit the leakage-safe encoder on the SAME train split as training and emit one row per
    geography plus a GLOBAL_KEY fallback row, timestamped for Feast."""
    df = load_raw(cfg.data_path)
    X, y = df[INPUT_COLUMNS], df["turnover"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    geo_frame = pd.DataFrame(
        {"Geography": X_train["Geography"].to_numpy(), "turnover": y_train.to_numpy()}
    )
    encoder = GeographyChurnRateEncoder().fit(geo_frame)

    rows = [
        {"Geography": geo, FEATURE_NAME: float(rate)}
        for geo, rate in encoder.mapping_.items()
    ]
    rows.append({"Geography": GLOBAL_KEY, FEATURE_NAME: float(encoder.global_rate_)})
    frame = pd.DataFrame(rows)
    frame["event_timestamp"] = _EVENT_TS
    return frame


def materialize(cfg: Settings = settings) -> None:
    """Write the offline parquet, apply the definitions and materialize into the online store."""
    frame = build_offline_frame(cfg)
    offline_path = Path(cfg.feast_offline_path)
    offline_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(offline_path, index=False)

    store = build_store(cfg)
    store.apply([geography, build_feature_view(cfg)])
    store.materialize_incremental(end_date=_MATERIALIZE_END)


def main() -> None:
    materialize(settings)
    print(f"materialized feature store at {settings.feast_repo_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_feature_store.py::test_build_offline_frame_matches_encoder -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/churn/feature_store/materialize.py tests/test_feature_store.py
git commit -m "feat: build Feast offline frame from the train split and materialize"
```

---

## Task 4: Online read with global fallback + consistency proof

**Files:**
- Create: `src/churn/feature_store/store.py`
- Test: `tests/test_feature_store.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_feature_store.py`:

```python
@pytest.fixture(scope="module")
def materialized(tmp_path_factory):
    from churn.feature_store.materialize import materialize

    repo = tmp_path_factory.mktemp("feast")
    cfg = Settings(data_path=CSV_PATH, feast_repo_path=str(repo / "feature_repo"))
    materialize(cfg)
    return cfg


def test_materialize_end_to_end(materialized):
    from churn.feature_store.store import get_geography_churn_rate

    rates = get_geography_churn_rate(
        ["Minas Gerais", "Rio de Janeiro", "Sao Paulo"], cfg=materialized
    )
    assert set(rates) == {"Minas Gerais", "Rio de Janeiro", "Sao Paulo"}
    for value in rates.values():
        assert 0.0 < value < 1.0


def test_online_matches_training_encoder(materialized):
    from churn.feature_store.store import get_geography_churn_rate

    encoder = _train_encoder(materialized)
    served = get_geography_churn_rate(list(encoder.mapping_), cfg=materialized)
    for geo, rate in encoder.mapping_.items():
        assert served[geo] == pytest.approx(rate, abs=1e-6)


def test_unseen_geography_falls_back_to_global(materialized):
    from churn.feature_store.store import get_geography_churn_rate

    encoder = _train_encoder(materialized)
    served = get_geography_churn_rate(["Bahia"], cfg=materialized)
    assert served["Bahia"] == pytest.approx(encoder.global_rate_, abs=1e-6)


def test_missing_store_raises(tmp_path):
    from churn.feature_store.store import FeatureStoreUnavailable, get_geography_churn_rate

    cfg = _cfg(tmp_path)
    with pytest.raises(FeatureStoreUnavailable):
        get_geography_churn_rate(["Sao Paulo"], cfg=cfg)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_feature_store.py::test_missing_store_raises -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn.feature_store.store'`

- [ ] **Step 3: Implement `store.py`**

Create `src/churn/feature_store/store.py`:

```python
"""Online serving: read geography_churn_rate by entity key, with global fallback."""

from pathlib import Path

from churn.config import Settings, settings
from churn.feature_store.definitions import (
    FEATURE_NAME,
    FEATURE_VIEW_NAME,
    GLOBAL_KEY,
    build_store,
)


class FeatureStoreUnavailable(RuntimeError):
    """Raised when the Feast repo has not been materialized yet."""


def get_geography_churn_rate(
    geographies: list[str], cfg: Settings = settings
) -> dict[str, float]:
    """Return the online-served churn rate per geography; unseen geography -> global fallback."""
    if not Path(cfg.feast_registry_path).exists():
        raise FeatureStoreUnavailable(
            f"Feast registry not found at {cfg.feast_registry_path}. "
            "Run `make feast-materialize` first."
        )

    store = build_store(cfg)
    wanted = list(dict.fromkeys([*geographies, GLOBAL_KEY]))
    response = store.get_online_features(
        features=[f"{FEATURE_VIEW_NAME}:{FEATURE_NAME}"],
        entity_rows=[{"Geography": g} for g in wanted],
    ).to_dict()

    values = dict(zip(response["Geography"], response[FEATURE_NAME]))
    global_rate = values.get(GLOBAL_KEY)
    result: dict[str, float] = {}
    for geo in geographies:
        rate = values.get(geo)
        result[geo] = float(rate) if rate is not None else float(global_rate)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_feature_store.py -v`
Expected: PASS (all feature-store tests so far, including the module-scoped `materialized` fixture)

- [ ] **Step 5: Commit**

```bash
git add src/churn/feature_store/store.py tests/test_feature_store.py
git commit -m "feat: serve geography_churn_rate online with global fallback"
```

---

## Task 5: API endpoint for online serving

**Files:**
- Modify: `src/churn/serving/api.py`
- Test: `tests/test_feature_store.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_feature_store.py`:

```python
def test_feature_endpoint_serves_rate(materialized):
    from churn.serving.api import app, get_feature_cfg

    app.dependency_overrides[get_feature_cfg] = lambda: materialized
    try:
        response = TestClient(app).get("/features/geography/Sao Paulo")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    body = response.json()
    assert body["geography"] == "Sao Paulo"
    assert 0.0 < body["geography_churn_rate"] < 1.0


def test_feature_endpoint_503_without_store(tmp_path):
    from churn.serving.api import app, get_feature_cfg

    cfg = _cfg(tmp_path)
    app.dependency_overrides[get_feature_cfg] = lambda: cfg
    try:
        response = TestClient(app).get("/features/geography/Sao Paulo")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_feature_store.py::test_feature_endpoint_503_without_store -v`
Expected: FAIL with `ImportError: cannot import name 'get_feature_cfg' from 'churn.serving.api'`

- [ ] **Step 3: Implement the endpoint**

In `src/churn/serving/api.py`, add this import next to the existing `churn` imports (below the `from churn.serving.schema import ...` line):

```python
from churn.feature_store.store import FeatureStoreUnavailable, get_geography_churn_rate
```

Then append at the end of the file:

```python
def get_feature_cfg() -> Settings:
    """Settings used by the feature-store endpoint (overridable in tests)."""
    return settings


@app.get("/features/geography/{geography}")
def geography_feature(geography: str, cfg: Settings = Depends(get_feature_cfg)) -> dict:  # noqa: B008
    """Serve the online-materialized churn rate for a geography (503 if not materialized)."""
    try:
        rate = get_geography_churn_rate([geography], cfg=cfg)[geography]
    except FeatureStoreUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"geography": geography, "geography_churn_rate": rate}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_feature_store.py -v`
Expected: PASS (endpoint tests included)

- [ ] **Step 5: Commit**

```bash
git add src/churn/serving/api.py tests/test_feature_store.py
git commit -m "feat: add lazy GET /features/geography endpoint"
```

---

## Task 6: Gitignore, Makefile target, and full verification

**Files:**
- Modify: `.gitignore`
- Modify: `Makefile`

- [ ] **Step 1: Ignore generated Feast artifacts**

Append to `.gitignore`:

```gitignore
# Feast local artifacts (generated)
feature_repo/
```

- [ ] **Step 2: Add the Makefile target**

In `Makefile`, add `feast-materialize` to the `.PHONY` line:

```makefile
.PHONY: help setup test lint format train serve feast-materialize docker-build docker-run
```

Add this target after the `train` target:

```makefile
feast-materialize: ## Popula e materializa a feature store Feast (offline -> online)
	uv run python -m churn.feature_store.materialize
```

In the `docker-run` target, add the feature_repo mount (a new `-v` line before `churn-api`):

```makefile
docker-run: ## Roda o container com o registry MLflow local montado (precisa de `make train` antes)
	docker run --rm -p 8000:8000 \
		-v "$(CURDIR)/mlflow.db:/app/mlflow.db" \
		-v "$(CURDIR)/mlruns:$(CURDIR)/mlruns" \
		-v "$(CURDIR)/feature_repo:$(CURDIR)/feature_repo" \
		churn-api
```

- [ ] **Step 3: Verify the materialize target runs end-to-end**

Run: `make feast-materialize`
Expected: exits 0 and prints `materialized feature store at feature_repo`; creates `feature_repo/registry.db`, `feature_repo/online_store.db`, `feature_repo/data/geo_churn_stats.parquet`.

- [ ] **Step 4: Confirm generated artifacts are NOT tracked by git**

Run: `git status --porcelain feature_repo`
Expected: empty output (the directory is gitignored).

- [ ] **Step 5: Run the full suite and lint**

Run: `uv run pytest -v`
Expected: PASS — the 42 pre-existing tests plus the new feature-store tests are all green.

Run: `uv run ruff check src tests`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add .gitignore Makefile
git commit -m "chore: ignore Feast artifacts and add feast-materialize target"
```

---

## Final review

After all tasks: dispatch a final code review over the whole branch, then use superpowers:finishing-a-development-branch to open the PR (`Closes #9`). Do **not** add a Claude co-author trailer to any commit or PR in this repo.

---

## Self-review notes (author)

- **Spec coverage:** §3.1 definitions → Task 2; §3.1 materialize → Task 3; §3.1 store → Task 4; §3.2 config → Task 1; §3.3 endpoint → Task 5; §3.4 gitignore → Task 6; §3.5 Makefile → Task 6; §6 tests 1–5 → Tasks 2/3/4/5; §5 error handling (503 + global fallback) → Tasks 4/5. Vertex mapping (§2) stays documentation-only per spec (README milestone).
- **No placeholders:** every code step contains full code; every run step has an exact command and expected result.
- **Type consistency:** `build_store`, `build_feature_view`, `build_offline_frame`, `materialize`, `get_geography_churn_rate`, `FeatureStoreUnavailable`, `get_feature_cfg`, and the constants `GLOBAL_KEY` / `FEATURE_NAME` / `FEATURE_VIEW_NAME` are named identically across definition and use.

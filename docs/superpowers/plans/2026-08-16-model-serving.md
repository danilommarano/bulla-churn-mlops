# Milestone 3 — Model Serving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve the registered `churn-model@production` over a small FastAPI REST API (`/health` + `/predict`), reusing the single `scoring.retention_score` rule, and package it in Docker — mirroring a Vertex AI Endpoint locally.

**Architecture:** FastAPI app loads the MLflow `@production` model once at startup (lifespan) via the configured tracking URI. `POST /predict` accepts one or many records validated by a Pydantic schema (which mirrors `INPUT_COLUMNS`, with aliases for the space-containing column names), builds a DataFrame, and runs the *same persisted pipeline* used in training — so `geography_churn_rate`/`age_bucket` are derived identically with zero train/serve skew and no Feast dependency yet. Docker packages the app; the local registry (`mlflow.db` + `mlruns/`) is mounted at runtime.

**Tech Stack:** FastAPI, uvicorn, Pydantic v2, MLflow 3.15 (local SQLite), scikit-learn, pytest + httpx (TestClient), Docker.

**Verified before writing:** the schema→DataFrame→predict→`retention_score` flow, the `/health` + `/predict` (single, batch, null-optional, 422-on-invalid) behaviour via `TestClient` with a dependency override, and that `mlflow.sklearn.load_model("models:/churn-model@production")` loads AND is directory-portable (loads from a copied `mlflow.db`+`mlruns/` in a different CWD). Docker is available (29.5.1) so the Docker task builds + smoke-tests for real.

---

## File Structure

- `src/churn/serving/__init__.py` — **create**: package marker.
- `src/churn/serving/schema.py` — **create**: `CustomerFeatures` (request) + `Prediction` (response) Pydantic models.
- `src/churn/serving/api.py` — **create**: FastAPI `app`, `load_production_model`, `get_model` dependency, `/health`, `/predict`.
- `tests/test_api.py` — **create**: endpoint tests (TestClient + dependency override) + one registry round-trip integration test.
- `Makefile` — **modify**: add `serve` target.
- `docker/Dockerfile` — **create**.
- `.dockerignore` — **create**: keep the build context small (exclude venv, registry, git, study folder).

`.gitignore` already ignores `mlruns/`, `mlflow.db`, `mlartifacts/`.

---

## Task 0: Dependencies — ALREADY DONE ✅

**Context (do not redo):** `fastapi`, `uvicorn[standard]` (runtime) and `httpx` (dev, for `TestClient`) were added and committed as `build: add FastAPI, uvicorn and httpx for model serving` (commit `825e0f7`). Versions resolved: fastapi 0.141.1, uvicorn 0.52.3, httpx 0.28.1. **Start implementation at Task 1.**

---

## Task 1: Request/response schema

**Files:**
- Create: `src/churn/serving/__init__.py`
- Create: `src/churn/serving/schema.py`
- Test: `tests/test_api.py` (schema-only tests in this task; endpoint tests added in Task 2)

**What it does:** `CustomerFeatures` mirrors `INPUT_COLUMNS`. Pydantic field names cannot contain spaces, so the three space-named columns use `Field(alias=...)` with `populate_by_name=True`, and `Balance`/`EstimatedSalary` are optional (nullable in the data contract; the pipeline fills them). `Prediction` is the per-record response.

- [ ] **Step 1: Write the failing test**

Create `tests/test_api.py`:

```python
import pytest
from pydantic import ValidationError

from churn.serving.schema import CustomerFeatures, Prediction

_VALID = {
    "CreditScore": 650,
    "Age": 40,
    "Balance": 1000.0,
    "EstimatedSalary": 50000.0,
    "Tenure": 5,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "Satisfaction Score": 3,
    "Point Earned": 500,
    "Geography": "Sao Paulo",
    "Gender": "Male",
    "Card Type": "GOLD",
}


def test_customer_features_accepts_aliased_column_names():
    c = CustomerFeatures(**_VALID)
    dumped = c.model_dump(by_alias=True)
    # the space-containing names round-trip via aliases
    assert dumped["Satisfaction Score"] == 3
    assert dumped["Point Earned"] == 500
    assert dumped["Card Type"] == "GOLD"


def test_customer_features_optional_balance_may_be_null():
    c = CustomerFeatures(**{**_VALID, "Balance": None, "EstimatedSalary": None})
    assert c.model_dump(by_alias=True)["Balance"] is None


def test_customer_features_rejects_missing_required_field():
    bad = {k: v for k, v in _VALID.items() if k != "CreditScore"}
    with pytest.raises(ValidationError):
        CustomerFeatures(**bad)


def test_prediction_shape():
    p = Prediction(turnover_pred=1, prob_churn=0.7, score_retencao=3)
    assert p.turnover_pred == 1
    assert p.score_retencao == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn.serving'`.

- [ ] **Step 3: Implement the package marker and schema**

Create `src/churn/serving/__init__.py`:

```python
"""Serving subpackage: FastAPI app, request/response schema and model loading."""
```

Create `src/churn/serving/schema.py`:

```python
"""Request/response contracts for the scoring API (Pydantic v2)."""

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """One customer record, mirroring the raw INPUT_COLUMNS the pipeline expects.

    The three columns whose names contain spaces use aliases so they can be sent
    with their exact CSV names; Balance/EstimatedSalary are optional (the pipeline
    fills nulls with 0).
    """

    model_config = ConfigDict(populate_by_name=True)

    CreditScore: int
    Age: int
    Balance: float | None = None
    EstimatedSalary: float | None = None
    Tenure: int
    NumOfProducts: int
    HasCrCard: int
    satisfaction_score: int = Field(alias="Satisfaction Score")
    point_earned: int = Field(alias="Point Earned")
    Geography: str
    Gender: str
    card_type: str = Field(alias="Card Type")


class Prediction(BaseModel):
    """One scored customer."""

    turnover_pred: int
    prob_churn: float
    score_retencao: int
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/churn/serving/__init__.py src/churn/serving/schema.py tests/test_api.py
git commit -m "feat: add serving request/response schema"
```

IMPORTANT commit rule for this repo: do NOT add any "Co-Authored-By: Claude" or Claude credit trailer.

---

## Task 2: FastAPI app — endpoints + model loading

**Files:**
- Create: `src/churn/serving/api.py`
- Test: `tests/test_api.py` (append endpoint + loader tests)

**What it does:** `load_production_model(cfg)` sets the tracking URI and loads `models:/<name>@<alias>` from the registry. The lifespan handler loads it once into `app.state.model`. `get_model` is a dependency returning that model (503 if absent). `/health` is a pure liveness check (always 200). `/predict` validates records, builds a DataFrame in `INPUT_COLUMNS` order, runs the persisted pipeline, and maps the probability of *staying* (`predict_proba[:, 0]`) through `retention_score` (single source of the rule, bug #9). Endpoint tests override `get_model` with a freshly-fitted pipeline (fast, no MLflow); the loader gets one real train→load→predict integration test.

**Testability note:** `TestClient(app)` is used WITHOUT a `with` block in the unit tests, so the lifespan does NOT run and no model is loaded from disk — the dependency override supplies the model instead. The one integration test calls `load_production_model` directly.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_api.py` (add these imports at the top of the file, then the tests at the bottom):

```python
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from churn.data import load_raw
from churn.features.builder import INPUT_COLUMNS
from churn.serving.api import app, get_model, load_production_model
from churn.training.pipeline import build_pipeline

CSV_PATH = str(Path(__file__).resolve().parents[1] / "Customer-Churn-Records.csv")


@pytest.fixture(scope="module")
def fitted_model():
    df = load_raw(CSV_PATH)
    X, y = df[INPUT_COLUMNS], df["turnover"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return build_pipeline().fit(X_train, y_train)


@pytest.fixture
def client(fitted_model):
    app.dependency_overrides[get_model] = lambda: fitted_model
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_is_200():
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_predict_single_record(client):
    r = client.post("/predict", json=[_VALID])
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    rec = body[0]
    assert rec["turnover_pred"] in (0, 1)
    assert 0.0 <= rec["prob_churn"] <= 1.0
    assert 0 <= rec["score_retencao"] <= 10


def test_predict_batch(client):
    r = client.post("/predict", json=[_VALID, _VALID, _VALID])
    assert r.status_code == 200
    assert len(r.json()) == 3


def test_predict_rejects_invalid_payload(client):
    bad = {k: v for k, v in _VALID.items() if k != "CreditScore"}
    r = client.post("/predict", json=[bad])
    assert r.status_code == 422


def test_predict_handles_null_optional_fields(client):
    r = client.post("/predict", json=[{**_VALID, "Balance": None, "EstimatedSalary": None}])
    assert r.status_code == 200


def test_predict_requires_a_loaded_model():
    # no dependency override, lifespan not run -> app.state.model missing -> 503
    r = TestClient(app).post("/predict", json=[_VALID])
    assert r.status_code == 503


def test_load_production_model_round_trips(tmp_path):
    from churn.config import Settings
    from churn.training.train import train

    cfg = Settings(
        data_path=CSV_PATH,
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        mlflow_experiment="serve-test",
        model_name="churn-model-serve-test",
        model_alias="production",
    )
    train(cfg)
    model = load_production_model(cfg)
    sample = load_raw(cfg.data_path)[INPUT_COLUMNS].head(3)
    assert len(model.predict(sample)) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL with `ImportError: cannot import name 'app' from 'churn.serving.api'` (module doesn't exist yet).

- [ ] **Step 3: Implement the app**

Create `src/churn/serving/api.py`:

```python
"""FastAPI scoring service: loads the @production model and serves /predict + /health."""

import warnings
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request

from churn.config import Settings, settings
from churn.features.builder import INPUT_COLUMNS
from churn.scoring import retention_score
from churn.serving.schema import CustomerFeatures, Prediction


def load_production_model(cfg: Settings = settings):
    """Load `models:/<model_name>@<model_alias>` from the MLflow registry."""
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mlflow.sklearn.load_model(f"models:/{cfg.model_name}@{cfg.model_alias}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_production_model(settings)
    yield
    app.state.model = None


app = FastAPI(title="Churn scoring API", lifespan=lifespan)


def get_model(request: Request):
    """Return the loaded pipeline, or 503 if the model was not loaded."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=list[Prediction])
def predict(records: list[CustomerFeatures], model=Depends(get_model)) -> list[Prediction]:
    frame = pd.DataFrame([r.model_dump(by_alias=True) for r in records])[INPUT_COLUMNS]
    proba = model.predict_proba(frame)
    preds = model.predict(frame)
    scores = retention_score(proba[:, 0])  # probability of STAYING -> 0..10
    return [
        Prediction(turnover_pred=int(pred), prob_churn=float(churn), score_retencao=int(score))
        for pred, churn, score in zip(preds, proba[:, 1], scores)
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py -v`
Expected: PASS (all: 4 schema + 6 endpoint/loader tests). The `test_load_production_model_round_trips` trains on the full 10k rows into a tmp registry; allow a few seconds.

- [ ] **Step 5: Run ruff and the full suite, then commit**

Run: `uv run ruff check src tests` (must pass) and `uv run pytest -q` (nothing regressed).

```bash
git add src/churn/serving/api.py tests/test_api.py
git commit -m "feat: add FastAPI scoring service loading the production model"
```

IMPORTANT: do NOT `git add` `mlflow.db`, `mlruns/`, or `mlartifacts/` (generated + git-ignored). Only add the two files listed. No Claude credit trailer.

---

## Task 3: Makefile `serve` target + Docker packaging

**Files:**
- Modify: `Makefile`
- Create: `.dockerignore`
- Create: `docker/Dockerfile`

**What it does:** `make serve` runs uvicorn locally against the local registry. The Dockerfile packages the API with uv; the registry (`mlflow.db` + `mlruns/`) is mounted at runtime (not baked into the image). Docker IS available on this machine, so this task builds the image and smoke-tests a running container for real.

- [ ] **Step 1: Add the `serve` target to the Makefile**

In `Makefile`, add `serve` to the `.PHONY` line so it reads:

```makefile
.PHONY: help setup test lint format train serve
```

Append this target after the existing `train` target (keep the tab indentation):

```makefile
serve: ## Sobe a API FastAPI (modelo @production do MLflow local)
	uv run uvicorn churn.serving.api:app --host 0.0.0.0 --port 8000
```

- [ ] **Step 2: Create `.dockerignore`**

Create `.dockerignore` with EXACTLY:

```
.venv/
.git/
estudo-local/
mlruns/
mlartifacts/
*.db
.pytest_cache/
.ruff_cache/
__pycache__/
.remember/
docs/
legacy/
notebooks/
tests/
```

- [ ] **Step 3: Create `docker/Dockerfile`**

Create `docker/Dockerfile` with EXACTLY:

```dockerfile
FROM python:3.12-slim

# uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Dependency layer (cached unless pyproject/lock change)
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# The MLflow registry (mlflow.db + mlruns/) is mounted at runtime, not baked in.
ENV CHURN_MLFLOW_TRACKING_URI=sqlite:///mlflow.db
EXPOSE 8000

CMD ["uv", "run", "--no-dev", "uvicorn", "churn.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 4: Build the image**

Run: `docker build -f docker/Dockerfile -t churn-api .`
Expected: builds successfully (the `uv sync` layer installs the runtime deps; the `churn` package builds from `src/`). If the build fails, read the error and fix the Dockerfile (common causes: a needed file not COPYed, or a `--no-dev` dep actually needed at runtime). Report the outcome.

- [ ] **Step 5: Smoke-test a running container**

First ensure a Production model exists locally (regenerate if needed): `make train`.
Then run the container with the registry mounted, and hit both endpoints:

```bash
docker run -d --name churn-api-smoke -p 8000:8000 \
  -v "$(pwd)/mlflow.db:/app/mlflow.db" \
  -v "$(pwd)/mlruns:/app/mlruns" \
  churn-api
sleep 8
curl -sf http://localhost:8000/health
curl -sf -X POST http://localhost:8000/predict -H 'Content-Type: application/json' \
  -d '[{"CreditScore":650,"Age":40,"Balance":1000.0,"EstimatedSalary":50000.0,"Tenure":5,"NumOfProducts":2,"HasCrCard":1,"Satisfaction Score":3,"Point Earned":500,"Geography":"Sao Paulo","Gender":"Male","Card Type":"GOLD"}]'
docker logs churn-api-smoke | tail -20
docker rm -f churn-api-smoke
```

Expected: `/health` returns `{"status":"ok"}`; `/predict` returns a JSON list with one object containing `turnover_pred`, `prob_churn`, `score_retencao`. If the container cannot load the model, check the volume mounts and that `make train` populated `mlflow.db`+`mlruns/`. Report the actual curl output.

- [ ] **Step 6: Commit**

```bash
git add Makefile .dockerignore docker/Dockerfile
git commit -m "feat: add make serve target and Docker packaging for the API"
```

IMPORTANT: no Claude credit trailer.

---

## Final verification (after all tasks)

```bash
uv run pytest -q          # all Milestone 1/2/3 tests green
uv run ruff check .       # All checks passed!
```

Then use **superpowers:finishing-a-development-branch** to open the PR referencing `Closes #7`.

---

## Self-Review

**Spec coverage (design phase 5 = "Serving — FastAPI + scoring + Docker"):**
- FastAPI `/predict` (single + batch) + `/health` → Task 2 ✅
- Loads the Registry `@production` model → Task 2 (`load_production_model` + lifespan) ✅
- Returns `turnover_pred` + `score_retencao` via the single `retention_score` rule → Task 2 (bug #9 stays closed) ✅
- Pydantic validates the payload → Task 1 (`CustomerFeatures`; 422 on invalid) ✅
- No train/serve skew: the same persisted pipeline derives the features from the request → Task 2 (no Feast needed yet) ✅
- Docker packaging → Task 3 (Dockerfile + real build + container smoke test) ✅
- `make serve` → Task 3 ✅

**Out of scope (correctly deferred):** Feast online/offline store (Milestone: feature store), KFP DAG, Evidently monitoring, CI/CD, TensorFlow variant doc, README final. `docker-compose`/an MLflow *server* are intentionally omitted — the SQLite backend + mounted registry cover local serving; compose can arrive with monitoring if warranted.

**Placeholder scan:** none — every code step carries complete code.

**Type consistency:** `CustomerFeatures.model_dump(by_alias=True)` yields the exact CSV column names; `pd.DataFrame(...)[INPUT_COLUMNS]` (from `features/builder.py`) reindexes to the pipeline's expected order. `load_production_model(cfg)`/`get_model`/`app` are the exact names imported by `tests/test_api.py`. `Prediction(turnover_pred, prob_churn, score_retencao)` matches the response asserted in the tests. `_VALID` (Task 1) is reused by Task 2's endpoint tests in the same file.

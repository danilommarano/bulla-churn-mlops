from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sklearn.model_selection import train_test_split

from churn.data import load_raw
from churn.features.builder import INPUT_COLUMNS
from churn.serving.api import app, get_model, load_production_model
from churn.serving.schema import CustomerFeatures, Prediction
from churn.training.pipeline import build_pipeline

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


def test_predict_empty_list_returns_empty(client):
    r = client.post("/predict", json=[])
    assert r.status_code == 200
    assert r.json() == []


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

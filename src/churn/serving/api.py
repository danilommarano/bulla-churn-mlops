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
def predict(records: list[CustomerFeatures], model=Depends(get_model)) -> list[Prediction]:  # noqa: B008
    if not records:
        return []
    frame = pd.DataFrame([r.model_dump(by_alias=True) for r in records])[INPUT_COLUMNS]
    proba = model.predict_proba(frame)
    preds = model.predict(frame)
    scores = retention_score(proba[:, 0])  # probability of STAYING -> 0..10
    return [
        Prediction(turnover_pred=int(pred), prob_churn=float(churn), score_retencao=int(score))
        for pred, churn, score in zip(preds, proba[:, 1], scores)
    ]

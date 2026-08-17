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
    assert geography.join_key == "Geography"
    fv = build_feature_view(cfg)
    assert fv.name == FEATURE_VIEW_NAME == "geo_churn_stats"
    assert FEATURE_NAME in [f.name for f in fv.features]
    assert GLOBAL_KEY == "__global__"

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

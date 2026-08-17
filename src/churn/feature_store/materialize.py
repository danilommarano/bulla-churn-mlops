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

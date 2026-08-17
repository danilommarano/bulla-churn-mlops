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

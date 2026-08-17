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
    if global_rate is None:
        raise FeatureStoreUnavailable(
            f"Global fallback key '{GLOBAL_KEY}' missing from the online store. "
            "The store may be partially materialized — re-run `make feast-materialize`."
        )
    result: dict[str, float] = {}
    for geo in geographies:
        rate = values.get(geo)
        result[geo] = float(rate) if rate is not None else float(global_rate)
    return result

"""Typed pipeline configuration, read from environment variables / .env."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pipeline parameters. Environment variables use the CHURN_ prefix."""

    model_config = SettingsConfigDict(env_prefix="CHURN_", env_file=".env", extra="ignore")

    data_path: str = "Customer-Churn-Records.csv"
    random_state: int = 42
    test_size: float = 0.2
    n_age_bins: int = 5
    min_roc_auc: float = 0.70  # promotion gate: register_model moves @production only if roc_auc >= this

    # MLflow (local, no server) + model registry
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "churn"
    model_name: str = "churn-model"
    model_alias: str = "production"

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

    # Model monitoring (Vertex AI Model Monitoring equivalent)
    reports_dir: str = "reports"
    drift_threshold: float = 0.3  # Jensen-Shannon per-feature, Vertex default parity

    @property
    def monitoring_report_path(self) -> str:
        return str(Path(self.reports_dir) / "drift.html")

    @property
    def monitoring_metrics_path(self) -> str:
        return str(Path(self.reports_dir) / "metrics.json")


settings = Settings()

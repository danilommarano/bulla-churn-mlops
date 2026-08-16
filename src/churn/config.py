"""Typed pipeline configuration, read from environment variables / .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pipeline parameters. Environment variables use the CHURN_ prefix."""

    model_config = SettingsConfigDict(env_prefix="CHURN_", env_file=".env", extra="ignore")

    data_path: str = "Customer-Churn-Records.csv"
    random_state: int = 42
    test_size: float = 0.2
    n_age_bins: int = 5

    # MLflow (local, no server) + model registry
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "churn"
    model_name: str = "churn-model"
    model_alias: str = "production"


settings = Settings()

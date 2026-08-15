"""Configuração tipada da pipeline, lida de variáveis de ambiente / .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Parâmetros da pipeline. Prefixo CHURN_ nas variáveis de ambiente."""

    model_config = SettingsConfigDict(env_prefix="CHURN_", env_file=".env", extra="ignore")

    data_path: str = "Customer-Churn-Records.csv"
    random_state: int = 42
    test_size: float = 0.2


settings = Settings()

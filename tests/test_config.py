from churn.config import Settings


def test_defaults():
    s = Settings()
    assert s.data_path == "Customer-Churn-Records.csv"
    assert s.random_state == 42
    assert s.test_size == 0.2


def test_env_override(monkeypatch):
    monkeypatch.setenv("CHURN_RANDOM_STATE", "7")
    monkeypatch.setenv("CHURN_TEST_SIZE", "0.3")
    s = Settings()
    assert s.random_state == 7
    assert s.test_size == 0.3


def test_mlflow_and_model_defaults():
    s = Settings()
    assert s.n_age_bins == 5
    assert s.mlflow_tracking_uri == "sqlite:///mlflow.db"
    assert s.mlflow_experiment == "churn"
    assert s.model_name == "churn-model"
    assert s.model_alias == "production"

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

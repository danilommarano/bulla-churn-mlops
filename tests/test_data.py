from churn.data import load_raw


def test_load_raw_validates_and_loads():
    df = load_raw("Customer-Churn-Records.csv")
    assert len(df) == 10000
    assert "turnover" in df.columns
    # o schema garante domínio: nenhuma geografia fora das 3 conhecidas
    assert set(df["Geography"].unique()) <= {"Minas Gerais", "Rio de Janeiro", "Sao Paulo"}

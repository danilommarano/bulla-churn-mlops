from pathlib import Path

from churn.data import load_raw

CSV_PATH = str(Path(__file__).resolve().parents[1] / "Customer-Churn-Records.csv")


def test_load_raw_validates_and_loads():
    df = load_raw(CSV_PATH)
    assert len(df) == 10000
    assert "turnover" in df.columns
    # the schema enforces the domain: no geography outside the 3 known ones
    assert set(df["Geography"].unique()) <= {
        "Minas Gerais",
        "Rio de Janeiro",
        "Sao Paulo",
    }

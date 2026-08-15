import pandas as pd
import pytest
from pandera.errors import SchemaError

from churn.schema import validate_raw


def _valid_row() -> dict:
    return {
        "RowNumber": 1,
        "CustomerId": 15634602,
        "Surname": "Hargrave",
        "CreditScore": 619,
        "Geography": "Sao Paulo",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 2,
        "Balance": 0.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 101348.88,
        "turnover": 1,
        "Complain": 1,
        "Satisfaction Score": 2,
        "Card Type": "DIAMOND",
        "Point Earned": 464,
    }


def test_valid_frame_passes():
    df = pd.DataFrame([_valid_row()])
    out = validate_raw(df)
    assert len(out) == 1
    assert list(out.columns) == list(df.columns)


def test_unknown_geography_rejected():
    row = _valid_row()
    row["Geography"] = "Lisboa"
    with pytest.raises(SchemaError):
        validate_raw(pd.DataFrame([row]))


def test_target_out_of_domain_rejected():
    row = _valid_row()
    row["turnover"] = 2
    with pytest.raises(SchemaError):
        validate_raw(pd.DataFrame([row]))


def test_impossible_age_rejected():
    row = _valid_row()
    row["Age"] = -5
    with pytest.raises(SchemaError):
        validate_raw(pd.DataFrame([row]))


def test_missing_column_rejected():
    row = _valid_row()
    del row["CreditScore"]
    with pytest.raises(SchemaError):
        validate_raw(pd.DataFrame([row]))


def test_nullable_balance_accepted():
    row = _valid_row()
    row["Balance"] = None
    out = validate_raw(pd.DataFrame([row]))
    assert len(out) == 1

"""Raw CSV data contract (Pandera). Rejects out-of-spec data before training."""

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

GEOGRAPHIES = ["Minas Gerais", "Rio de Janeiro", "Sao Paulo"]
GENDERS = ["Male", "Female"]
CARD_TYPES = ["DIAMOND", "GOLD", "PLATINUM", "SILVER"]

RAW_SCHEMA = DataFrameSchema(
    {
        "RowNumber": Column(int, Check.ge(1)),
        "CustomerId": Column(int),
        "Surname": Column(str),
        "CreditScore": Column(int, Check.in_range(300, 900)),
        "Geography": Column(str, Check.isin(GEOGRAPHIES)),
        "Gender": Column(str, Check.isin(GENDERS)),
        "Age": Column(int, Check.in_range(18, 120)),
        "Tenure": Column(int, Check.in_range(0, 10)),
        "Balance": Column(float, Check.ge(0), nullable=True),
        "NumOfProducts": Column(int, Check.in_range(1, 4)),
        "HasCrCard": Column(int, Check.isin([0, 1])),
        "IsActiveMember": Column(int, Check.isin([0, 1])),
        "EstimatedSalary": Column(float, Check.ge(0), nullable=True),
        "turnover": Column(int, Check.isin([0, 1])),
        "Complain": Column(int, Check.isin([0, 1])),
        "Satisfaction Score": Column(int, Check.in_range(1, 5)),
        "Card Type": Column(str, Check.isin(CARD_TYPES)),
        "Point Earned": Column(int, Check.ge(0)),
    },
    strict=False,  # tolerate extra columns (e.g., index)
    coerce=True,
)


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the raw DataFrame against RAW_SCHEMA. Raises SchemaError if invalid."""
    return RAW_SCHEMA.validate(df, lazy=False)

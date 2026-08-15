"""CSV loading with schema validation at the entry point."""

import pandas as pd

from churn.schema import validate_raw


def load_raw(path: str) -> pd.DataFrame:
    """Read the CSV and validate it against the contract. Raises SchemaError if out of spec."""
    df = pd.read_csv(path)
    return validate_raw(df)

"""Carregamento do CSV com validação de schema na porta de entrada."""

import pandas as pd

from churn.schema import validate_raw


def load_raw(path: str) -> pd.DataFrame:
    """Lê o CSV e valida contra o contrato. Lança SchemaError se algo estiver fora do padrão."""
    df = pd.read_csv(path)
    return validate_raw(df)

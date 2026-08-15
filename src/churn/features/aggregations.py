"""Encoder da taxa de churn por geografia — aprendido SÓ no treino (anti-leakage, bug #1)."""

import pandas as pd


class GeographyChurnRateEncoder:
    """Aprende a taxa média de `turnover` por `Geography` no conjunto de treino e a aplica.

    - `fit(df_train)`: memoriza a média do alvo por grupo e a taxa global (fallback).
    - `transform(df)`: mapeia cada linha para a taxa aprendida; geografia não vista → taxa global.

    Como o mapeamento vem exclusivamente do treino, o conjunto de teste/produção nunca
    influencia o valor da feature — não há vazamento do rótulo.
    """

    def __init__(self, geography_col: str = "Geography", target_col: str = "turnover"):
        self.geography_col = geography_col
        self.target_col = target_col

    def fit(self, df: pd.DataFrame) -> "GeographyChurnRateEncoder":
        self.mapping_ = df.groupby(self.geography_col)[self.target_col].mean().to_dict()
        self.global_rate_ = float(df[self.target_col].mean())
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        return df[self.geography_col].map(self.mapping_).fillna(self.global_rate_)

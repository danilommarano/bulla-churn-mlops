"""Deterministic drift simulation for demonstrating monitoring detection.

Shifts a few numeric feature distributions by fixed amounts so a monitoring run
on the perturbed holdout produces visible input + score drift. Categorical
columns and the target are preserved. No global RNG; fully deterministic.
"""

import pandas as pd

# Fixed additive/multiplicative shifts applied to raw numeric features present.
_SHIFTS = {
    "Age": lambda s: s + 15,
    "Balance": lambda s: s * 1.5,
    "CreditScore": lambda s: s - 100,
    "EstimatedSalary": lambda s: s * 1.3,
}


def simulate_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with selected numeric columns shifted. Pure/deterministic."""
    out = df.copy()
    for column, shift in _SHIFTS.items():
        if column in out.columns:
            out[column] = shift(out[column])
    return out

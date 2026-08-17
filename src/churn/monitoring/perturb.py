"""Deterministic drift simulation for demonstrating monitoring detection.

Shifts a few numeric feature distributions by fixed amounts so a monitoring run
on the perturbed holdout produces visible input + score drift. Categorical
columns and the target are preserved. No global RNG; fully deterministic.
"""

import pandas as pd

# Fixed additive/multiplicative shifts applied to raw numeric features present.
# Chosen strong enough that several columns cross the 0.3 Jensen-Shannon threshold,
# so the `make monitor-drift` demo fails the gate with a comfortable margin (not a
# knife-edge that a resample or an Evidently patch could silently flip to PASS).
_SHIFTS = {
    "Age": lambda s: s + 20,
    "Balance": lambda s: s * 2.0,
    "CreditScore": lambda s: s - 150,
    "EstimatedSalary": lambda s: s * 1.6,
    "Point Earned": lambda s: s * 1.6,
}


def simulate_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with selected numeric columns shifted. Pure/deterministic."""
    out = df.copy()
    for column, shift in _SHIFTS.items():
        if column in out.columns:
            out[column] = shift(out[column])
    return out

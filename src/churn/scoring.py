"""Single retention-scoring rule: probability of STAYING -> integer 0-10.

Imported by training, batch inference and the API to avoid duplicating the rule (fixes bug #9).
"""

import numpy as np


def retention_score(prob_stay):
    """Convert the probability that the customer stays (0..1) into an integer 0-10 score.

    Accepts a scalar or an array. Returns int (scalar) or an ndarray of int (array).
    """
    scores = np.round(np.asarray(prob_stay, dtype=float) * 10).astype(int)
    if scores.ndim == 0:
        return int(scores)
    return scores

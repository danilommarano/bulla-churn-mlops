"""Regra única de score de retenção: probabilidade de PERMANECER → inteiro 0–10.

Importada por treino, inferência batch e API, para não duplicar a regra (fecha bug #9).
"""

import numpy as np


def retention_score(prob_stay):
    """Converte a probabilidade de o cliente permanecer (0..1) em um score inteiro 0–10.

    Aceita escalar ou array. Retorna int (escalar) ou ndarray de int (array).
    """
    scores = np.round(np.asarray(prob_stay, dtype=float) * 10).astype(int)
    if scores.ndim == 0:
        return int(scores)
    return scores

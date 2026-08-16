import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from churn.training.evaluate import evaluate


def _perfectly_separable():
    X = pd.DataFrame({"x": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]})
    y = pd.Series([0, 0, 0, 1, 1, 1])
    return LogisticRegression().fit(X, y), X, y


def test_evaluate_returns_all_metrics():
    model, X, y = _perfectly_separable()
    m = evaluate(model, X, y)
    assert set(m) == {"roc_auc", "precision", "recall", "f1", "accuracy", "confusion_matrix"}


def test_metrics_are_in_range_and_high_on_separable_data():
    model, X, y = _perfectly_separable()
    m = evaluate(model, X, y)
    for key in ("roc_auc", "precision", "recall", "f1", "accuracy"):
        assert 0.0 <= m[key] <= 1.0
    assert m["roc_auc"] == 1.0


def test_confusion_matrix_is_serializable_2x2():
    model, X, y = _perfectly_separable()
    m = evaluate(model, X, y)
    assert isinstance(m["confusion_matrix"], list)
    assert np.array(m["confusion_matrix"]).shape == (2, 2)

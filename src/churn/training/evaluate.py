"""Honest classification metrics for an imbalanced churn base (fixes bug #5)."""

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(model, X, y) -> dict:
    """Compute ROC-AUC, precision, recall, F1, accuracy and the confusion matrix.

    `confusion_matrix` is returned as a nested list so it is JSON/MLflow friendly.
    """
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }

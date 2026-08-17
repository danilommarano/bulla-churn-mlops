import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from churn.data import load_raw
from churn.features.builder import INPUT_COLUMNS
from churn.training.pipeline import build_pipeline

CSV_PATH = str(Path(__file__).resolve().parents[1] / "Customer-Churn-Records.csv")


def _split():
    df = load_raw(CSV_PATH)
    X, y = df[INPUT_COLUMNS], df["turnover"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def test_pipeline_fits_and_predicts():
    X_train, X_test, y_train, _y_test = _split()
    pipe = build_pipeline().fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert set(np.unique(pipe.predict(X_test))) <= {0, 1}


def test_pipeline_survives_persist_roundtrip():
    """The whole fitted pipeline is self-contained: reload gives identical output (bug #2)."""
    X_train, X_test, y_train, _ = _split()
    pipe = build_pipeline().fit(X_train, y_train)
    before = pipe.predict_proba(X_test)[:, 1]
    path = Path(tempfile.mkdtemp()) / "pipe.pkl"
    joblib.dump(pipe, path)
    after = joblib.load(path).predict_proba(X_test)[:, 1]
    assert np.allclose(before, after)


def test_pipeline_is_reproducible():
    """Same seed -> same predictions (bug #6)."""
    X_train, X_test, y_train, _ = _split()
    a = (
        build_pipeline(random_state=42)
        .fit(X_train, y_train)
        .predict_proba(X_test)[:, 1]
    )
    b = (
        build_pipeline(random_state=42)
        .fit(X_train, y_train)
        .predict_proba(X_test)[:, 1]
    )
    assert np.allclose(a, b)

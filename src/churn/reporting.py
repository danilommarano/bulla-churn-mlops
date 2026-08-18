"""Exporta as métricas do modelo @production para o site de apresentação.

Versiona o que o helper ad-hoc de staging fazia: reproduz o mesmo split do projeto,
avalia o @production com o `evaluate` canônico (sem duplicar lógica de métrica) e grava
o JSON consumido pelo build do site (`presentation/src/data/metrics.json`).
"""

import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from churn.config import Settings, settings
from churn.data import load_raw
from churn.features.builder import INPUT_COLUMNS
from churn.serving.api import load_production_model
from churn.training.evaluate import evaluate

DEFAULT_OUT = Path("presentation/src/data/metrics.json")


def export_production_metrics(cfg: Settings = settings, out_path: Path = DEFAULT_OUT) -> dict:
    """Avalia o @production no holdout de teste e grava as métricas em `out_path`."""
    model = load_production_model(cfg)
    df = load_raw(cfg.data_path)
    X, y = df[INPUT_COLUMNS], df["turnover"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    metrics = evaluate(model, X_test, y_test)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    metrics = export_production_metrics()
    print(json.dumps(metrics, indent=2))
    print(f"-> {DEFAULT_OUT}")

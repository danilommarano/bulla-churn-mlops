"""KFP local pipeline: orchestrate the churn training DAG (Vertex AI Pipelines equivalent).

Run end to end with `python -m churn.orchestration.dag` (make pipeline). Uses the local
SubprocessRunner so each component runs in the current venv — no cluster, no container images.
"""

from kfp import dsl, local
from kfp.dsl import Dataset, Input, Metrics, Model, Output

from churn.config import Settings, settings


@dsl.component
def prepare_data_op(output: Output[Dataset], data_path: str):
    from churn.orchestration.steps.prepare_data import prepare_data

    prepare_data(output.path, data_path)


@dsl.component
def split_data_op(
    dataset: Input[Dataset],
    train: Output[Dataset],
    test: Output[Dataset],
    test_size: float,
    random_state: int,
):
    from churn.orchestration.steps.split_data import split_data

    split_data(dataset.path, train.path, test.path, test_size, random_state)


@dsl.component
def train_model_op(
    train_set: Input[Dataset], model: Output[Model], random_state: int, n_age_bins: int
):
    from churn.orchestration.steps.train_model import train_model

    train_model(train_set.path, model.path, random_state, n_age_bins)


@dsl.component
def evaluate_model_op(
    model: Input[Model], test_set: Input[Dataset], metrics: Output[Metrics]
):
    from churn.orchestration.steps.evaluate_model import evaluate_model

    result = evaluate_model(model.path, test_set.path, metrics.path)
    for key in ("roc_auc", "precision", "recall", "f1", "accuracy"):
        metrics.log_metric(key, float(result[key]))


@dsl.component
def register_model_op(
    model: Input[Model],
    metrics: Input[Metrics],
    train_set: Input[Dataset],
    mlflow_tracking_uri: str,
    mlflow_experiment: str,
    model_name: str,
    model_alias: str,
    random_state: int,
    test_size: float,
    n_age_bins: int,
    min_roc_auc: float,
):
    from churn.config import Settings
    from churn.orchestration.steps.register_model import (
        register_model,
        require_promotion,
    )

    cfg = Settings(
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
        model_name=model_name,
        model_alias=model_alias,
        random_state=random_state,
        test_size=test_size,
        n_age_bins=n_age_bins,
        min_roc_auc=min_roc_auc,
    )
    result = register_model(model.path, metrics.path, train_set.path, cfg)
    require_promotion(result)


@dsl.pipeline(name="churn-training-pipeline")
def churn_training_pipeline(
    data_path: str,
    test_size: float,
    random_state: int,
    n_age_bins: int,
    mlflow_tracking_uri: str,
    mlflow_experiment: str,
    model_name: str,
    model_alias: str,
    min_roc_auc: float,
):
    prep = prepare_data_op(data_path=data_path)
    split = split_data_op(
        dataset=prep.outputs["output"], test_size=test_size, random_state=random_state
    )
    trained = train_model_op(
        train_set=split.outputs["train"],
        random_state=random_state,
        n_age_bins=n_age_bins,
    )
    ev = evaluate_model_op(
        model=trained.outputs["model"], test_set=split.outputs["test"]
    )
    register_model_op(
        model=trained.outputs["model"],
        metrics=ev.outputs["metrics"],
        train_set=split.outputs["train"],
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
        model_name=model_name,
        model_alias=model_alias,
        random_state=random_state,
        test_size=test_size,
        n_age_bins=n_age_bins,
        min_roc_auc=min_roc_auc,
    )


def run_local(cfg: Settings = settings, pipeline_root: str | None = None):
    """Initialize the local SubprocessRunner and execute the DAG end to end.

    `pipeline_root` overrides where KFP writes component artifacts (defaults to
    ./local_outputs). Note: local.init sets KFP-global state, so the last call wins.
    """
    init_kwargs = {
        "runner": local.SubprocessRunner(use_venv=False),
        "raise_on_error": True,
        "enable_caching": False,
    }
    if pipeline_root is not None:
        init_kwargs["pipeline_root"] = pipeline_root
    local.init(**init_kwargs)
    return churn_training_pipeline(
        data_path=cfg.data_path,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        n_age_bins=cfg.n_age_bins,
        mlflow_tracking_uri=cfg.mlflow_tracking_uri,
        mlflow_experiment=cfg.mlflow_experiment,
        model_name=cfg.model_name,
        model_alias=cfg.model_alias,
        min_roc_auc=cfg.min_roc_auc,
    )


def main() -> None:
    run_local()
    print("Pipeline finished. Check MLflow for the registered model version.")


if __name__ == "__main__":
    main()

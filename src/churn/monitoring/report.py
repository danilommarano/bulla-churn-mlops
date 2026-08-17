"""Build the Evidently monitoring report (input drift + score drift + quality).

Uses the Evidently 0.7 API. Data/score drift use Jensen-Shannon (Vertex AI Model
Monitoring parity); the per-feature threshold comes from cfg.drift_threshold.

Import paths verified against evidently 0.7.21:
  - Report, Dataset, DataDefinition, BinaryClassification: top-level `evidently`.
  - ValueDrift: `evidently.metrics`.
  - DataDriftPreset, ClassificationQuality: `evidently.presets`.
DataDefinition.classification takes a *list* of BinaryClassification objects.
DataDriftPreset exposes num_method/threshold (Jensen-Shannon parity); ValueDrift
takes a keyword `column`. report.run(...) is called with keyword current_/reference_.
"""

from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift
from evidently.presets import ClassificationQuality, DataDriftPreset

from churn.config import Settings
from churn.features.builder import RAW_CATEGORICAL, RAW_NUMERIC


def _dataset(frame, definition):
    return Dataset.from_pandas(frame, data_definition=definition)


def build_report(reference_df, current_df, cfg: Settings):
    """Run the monitoring report and return the Evidently snapshot."""
    definition = DataDefinition(
        numerical_columns=[*RAW_NUMERIC, "prob_churn"],
        categorical_columns=list(RAW_CATEGORICAL),
        classification=[
            BinaryClassification(
                target="turnover", prediction_probas="prob_churn", pos_label=1
            )
        ],
    )
    report = Report(
        [
            # Jensen-Shannon + cfg.drift_threshold on NUMERIC columns for Vertex AI
            # parity; DriftedColumnsCount.value.share feeds the quality gate. We set
            # num_threshold only (not the global `threshold`, which would also apply
            # 0.3 to the categorical p-value stattests — chi-square/Z-test — where a
            # 0.3 cutoff is semantically unrelated). Categorical columns keep
            # Evidently's default stattest/threshold; Vertex's L-infinity for
            # categoricals has no Evidently equivalent (documented gap in the spec).
            DataDriftPreset(num_method="jensenshannon", num_threshold=cfg.drift_threshold),
            ValueDrift(column="prob_churn"),
            ClassificationQuality(),
        ],
        include_tests=True,
    )
    reference = _dataset(reference_df, definition)
    current = _dataset(current_df, definition)
    return report.run(current_data=current, reference_data=reference)

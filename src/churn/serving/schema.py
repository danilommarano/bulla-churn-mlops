"""Request/response contracts for the scoring API (Pydantic v2)."""

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """One customer record, mirroring the raw INPUT_COLUMNS the pipeline expects.

    The three columns whose names contain spaces use aliases so they can be sent
    with their exact CSV names; Balance/EstimatedSalary are optional (the pipeline
    fills nulls with 0).
    """

    model_config = ConfigDict(populate_by_name=True)

    CreditScore: int
    Age: int
    Balance: float | None = None
    EstimatedSalary: float | None = None
    Tenure: int
    NumOfProducts: int
    HasCrCard: int
    satisfaction_score: int = Field(alias="Satisfaction Score")
    point_earned: int = Field(alias="Point Earned")
    Geography: str
    Gender: str
    card_type: str = Field(alias="Card Type")


class Prediction(BaseModel):
    """One scored customer."""

    turnover_pred: int
    prob_churn: float
    score_retencao: int

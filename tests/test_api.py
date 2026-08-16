import pytest
from pydantic import ValidationError

from churn.serving.schema import CustomerFeatures, Prediction

_VALID = {
    "CreditScore": 650,
    "Age": 40,
    "Balance": 1000.0,
    "EstimatedSalary": 50000.0,
    "Tenure": 5,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "Satisfaction Score": 3,
    "Point Earned": 500,
    "Geography": "Sao Paulo",
    "Gender": "Male",
    "Card Type": "GOLD",
}


def test_customer_features_accepts_aliased_column_names():
    c = CustomerFeatures(**_VALID)
    dumped = c.model_dump(by_alias=True)
    # the space-containing names round-trip via aliases
    assert dumped["Satisfaction Score"] == 3
    assert dumped["Point Earned"] == 500
    assert dumped["Card Type"] == "GOLD"


def test_customer_features_optional_balance_may_be_null():
    c = CustomerFeatures(**{**_VALID, "Balance": None, "EstimatedSalary": None})
    assert c.model_dump(by_alias=True)["Balance"] is None


def test_customer_features_rejects_missing_required_field():
    bad = {k: v for k, v in _VALID.items() if k != "CreditScore"}
    with pytest.raises(ValidationError):
        CustomerFeatures(**bad)


def test_prediction_shape():
    p = Prediction(turnover_pred=1, prob_churn=0.7, score_retencao=3)
    assert p.turnover_pred == 1
    assert p.score_retencao == 3

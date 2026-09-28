"""
Unit Tests for Machine Learning Model Pipeline.
"""

import os
import joblib
import pandas as pd
import pytest

from train import (
    MODEL_PIPELINE_PATH,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    train,
)


@pytest.fixture(scope="session")
def model_pipeline():
    if not os.path.exists(MODEL_PIPELINE_PATH):
        train()
    return joblib.load(MODEL_PIPELINE_PATH)


def test_pipeline_loading(model_pipeline):
    assert model_pipeline is not None
    assert hasattr(model_pipeline, "predict")
    assert hasattr(model_pipeline, "predict_proba")


def test_pipeline_prediction_format(model_pipeline):
    sample = pd.DataFrame([{
        "credit_score": 600,
        "country": "France",
        "gender": "Female",
        "age": 40,
        "tenure": 3,
        "balance": 50000.0,
        "num_of_products": 2,
        "has_credit_card": 1,
        "is_active_member": 1,
        "estimated_salary": 70000.0,
    }])

    pred = model_pipeline.predict(sample)
    proba = model_pipeline.predict_proba(sample)

    assert len(pred) == 1
    assert pred[0] in [0, 1]
    assert proba.shape == (1, 2)
    assert abs(proba[0].sum() - 1.0) < 1e-4


def test_pipeline_handles_unseen_categories(model_pipeline):
    # Tests that OneHotEncoder with handle_unknown='ignore' does not crash
    sample = pd.DataFrame([{
        "credit_score": 600,
        "country": "UnknownCountry",
        "gender": "Other",
        "age": 40,
        "tenure": 3,
        "balance": 50000.0,
        "num_of_products": 2,
        "has_credit_card": 1,
        "is_active_member": 1,
        "estimated_salary": 70000.0,
    }])

    pred = model_pipeline.predict(sample)
    assert len(pred) == 1

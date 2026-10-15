"""Tests for leakage-safe training helpers and manual input validation."""

import numpy as np
import pandas as pd

from src.modeling import build_prediction_frame, infer_task, train_model


def classification_frame() -> pd.DataFrame:
    ages = list(range(20, 60))
    return pd.DataFrame(
        {
            "age": [np.nan if index in {3, 17} else age for index, age in enumerate(ages)],
            "income": [age * 1_000 + (index % 3) * 250 for index, age in enumerate(ages)],
            "city": ["North" if index % 2 else "South" for index in range(40)],
            "approved": ["yes" if age >= 40 else "no" for age in ages],
        }
    )


def regression_frame() -> pd.DataFrame:
    size = list(range(20, 60))
    return pd.DataFrame(
        {
            "size": size,
            "zone": ["urban" if index % 2 else "rural" for index in range(40)],
            "price": [50_000 + value * 4_500 + (index % 2) * 8_000 for index, value in enumerate(size)],
        }
    )


def test_classification_training_returns_metrics_and_prediction_schema() -> None:
    frame = classification_frame()
    result = train_model(frame, "approved", ["age", "income", "city"], "classification", "Logistic Regression", test_size=0.25)

    assert infer_task(frame, "approved") == "classification"
    assert set(result.metrics) == {"Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 score (weighted)"}
    assert result.confusion is not None
    assert len(result.test_results) == result.test_rows
    request = build_prediction_frame({"age": 45, "income": 45_000, "city": "North"}, result.feature_schema)
    assert result.pipeline.predict(request)[0] in {"yes", "no"}


def test_regression_training_returns_expected_evaluation_metrics() -> None:
    frame = regression_frame()
    result = train_model(frame, "price", ["size", "zone"], "regression", "Linear Regression", test_size=0.25)

    assert infer_task(frame, "price") == "regression"
    assert set(result.metrics) == {"MAE", "RMSE", "R²"}
    assert result.confusion is None
    assert result.metrics["R²"] > 0.8


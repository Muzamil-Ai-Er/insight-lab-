"""Leakage-safe Scikit-learn model training and prediction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE

TaskType = Literal["classification", "regression"]


class ModelTrainingError(ValueError):
    """A friendly, anticipated error caused by unsuitable user data/settings."""


@dataclass
class TrainingResult:
    """Everything the UI needs after a successful fit."""

    pipeline: Pipeline
    task: TaskType
    algorithm: str
    target: str
    features: list[str]
    metrics: dict[str, float]
    test_results: pd.DataFrame
    confusion: pd.DataFrame | None
    classification_report: pd.DataFrame | None
    feature_schema: dict[str, dict[str, Any]]
    training_rows: int
    test_rows: int


def infer_task(frame: pd.DataFrame, target: str) -> TaskType:
    """Infer a conservative task type from the target's dtype and cardinality."""
    if target not in frame.columns:
        raise ModelTrainingError("Choose a target column from the current dataset.")
    series = frame[target].dropna()
    if series.empty:
        raise ModelTrainingError("The selected target contains no usable values.")
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"
    # Numeric labels with few distinct values are usually categories (such as 0/1).
    return "classification" if series.nunique() <= max(10, min(20, len(series) // 5)) else "regression"


def available_algorithms(task: TaskType) -> list[str]:
    """Expose only algorithms appropriate to the selected task."""
    if task == "classification":
        return ["Logistic Regression", "Decision Tree", "Random Forest"]
    return ["Linear Regression", "Random Forest Regressor"]


def _make_preprocessor(features: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = features.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [column for column in features.columns if column not in numeric_features]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )

    if not transformers:
        raise ModelTrainingError("Choose at least one usable numeric or categorical feature.")
    return ColumnTransformer(transformers=transformers, remainder="drop"), numeric_features, categorical_features


def _make_estimator(task: TaskType, algorithm: str):
    estimators: dict[tuple[TaskType, str], Any] = {
        ("classification", "Logistic Regression"): LogisticRegression(max_iter=1_500, random_state=RANDOM_STATE),
        ("classification", "Decision Tree"): DecisionTreeClassifier(random_state=RANDOM_STATE, min_samples_leaf=2),
        ("classification", "Random Forest"): RandomForestClassifier(
            n_estimators=250, random_state=RANDOM_STATE, n_jobs=-1, min_samples_leaf=1
        ),
        ("regression", "Linear Regression"): LinearRegression(),
        ("regression", "Random Forest Regressor"): RandomForestRegressor(
            n_estimators=250, random_state=RANDOM_STATE, n_jobs=-1, min_samples_leaf=1
        ),
    }
    try:
        return estimators[(task, algorithm)]
    except KeyError as error:
        raise ModelTrainingError("Choose an algorithm that matches the selected prediction task.") from error


def _can_stratify(target: pd.Series, test_size: float) -> bool:
    """Use stratification only when both split portions can contain every class."""
    counts = target.value_counts()
    test_rows = ceil(len(target) * test_size)
    train_rows = len(target) - test_rows
    return bool(len(counts) > 1 and counts.min() >= 2 and test_rows >= len(counts) and train_rows >= len(counts))


def _feature_schema(frame: pd.DataFrame, feature_names: list[str]) -> dict[str, dict[str, Any]]:
    """Create safe defaults and permitted categories for the prediction form."""
    schema: dict[str, dict[str, Any]] = {}
    for name in feature_names:
        series = frame[name]
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").dropna()
            default = float(values.median()) if not values.empty else 0.0
            schema[name] = {"kind": "numeric", "default": default}
        else:
            values = series.dropna().astype(str)
            categories = values.value_counts().index.tolist()[:100]
            if not categories:
                categories = ["Unknown"]
            schema[name] = {"kind": "categorical", "categories": categories, "default": categories[0]}
    return schema


def train_model(
    frame: pd.DataFrame,
    target: str,
    features: list[str],
    task: TaskType,
    algorithm: str,
    test_size: float = 0.20,
) -> TrainingResult:
    """Fit and evaluate a model with preprocessing learned only from training data.

    Rows missing the target are excluded because guessing the label would corrupt
    the evaluation. Feature imputation and one-hot encoding remain inside the
    pipeline, so they are fitted after the train/test split and cannot leak test
    information into training.
    """
    if target not in frame.columns:
        raise ModelTrainingError("Choose a valid target column.")
    selected_features = [name for name in features if name in frame.columns and name != target]
    if not selected_features:
        raise ModelTrainingError("Choose at least one feature different from the target.")
    if not 0.10 <= test_size <= 0.50:
        raise ModelTrainingError("Test size must be between 10% and 50%.")

    working = frame.loc[:, selected_features + [target]].dropna(subset=[target]).copy()
    if len(working) < 12:
        raise ModelTrainingError("At least 12 rows with a target value are required to train and evaluate a model.")

    features_frame = working.loc[:, selected_features]
    raw_target = working[target]
    if task == "classification":
        target_values = raw_target.astype(str)
        if target_values.nunique() < 2:
            raise ModelTrainingError("Classification needs at least two distinct target classes.")
        if target_values.nunique() > max(30, len(target_values) * 0.50):
            raise ModelTrainingError(
                "This target has too many unique labels for a reliable classification demo. "
                "Choose a lower-cardinality target or use regression for a continuous number."
            )
    else:
        target_values = pd.to_numeric(raw_target, errors="coerce")
        valid_target = target_values.notna()
        features_frame = features_frame.loc[valid_target].copy()
        target_values = target_values.loc[valid_target]
        if len(target_values) < 12:
            raise ModelTrainingError("Regression needs at least 12 numeric target values.")
        if target_values.nunique() < 4:
            raise ModelTrainingError("Regression needs a target with several different numeric values.")

    preprocessor, _, _ = _make_preprocessor(features_frame)
    estimator = _make_estimator(task, algorithm)
    stratify = target_values if task == "classification" and _can_stratify(target_values, test_size) else None
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features_frame,
            target_values,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=stratify,
        )
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
    except (ValueError, TypeError) as error:
        raise ModelTrainingError(
            f"The model could not be trained with these settings: {error}. "
            "Try a different target, fewer features, or a larger dataset."
        ) from error

    test_results = x_test.copy()
    test_results.insert(0, "Actual", y_test.to_numpy())
    test_results.insert(1, "Predicted", predictions)

    if task == "classification":
        labels = sorted(pd.unique(pd.concat([pd.Series(y_test), pd.Series(predictions)]).astype(str)))
        metrics = {
            "Accuracy": float(accuracy_score(y_test, predictions)),
            "Precision (weighted)": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
            "Recall (weighted)": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
            "F1 score (weighted)": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        }
        matrix = confusion_matrix(y_test, predictions, labels=labels)
        confusion = pd.DataFrame(matrix, index=pd.Index(labels, name="Actual"), columns=pd.Index(labels, name="Predicted"))
        report = pd.DataFrame(classification_report(y_test, predictions, output_dict=True, zero_division=0)).transpose()
    else:
        mse = float(mean_squared_error(y_test, predictions))
        metrics = {
            "MAE": float(mean_absolute_error(y_test, predictions)),
            "RMSE": float(np.sqrt(mse)),
            "R²": float(r2_score(y_test, predictions)),
        }
        confusion = None
        report = None

    return TrainingResult(
        pipeline=pipeline,
        task=task,
        algorithm=algorithm,
        target=target,
        features=selected_features,
        metrics=metrics,
        test_results=test_results.reset_index(drop=True),
        confusion=confusion,
        classification_report=report,
        feature_schema=_feature_schema(features_frame, selected_features),
        training_rows=len(x_train),
        test_rows=len(x_test),
    )


def build_prediction_frame(values: dict[str, Any], schema: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Validate manual values and return exactly one model-ready input row."""
    record: dict[str, Any] = {}
    for name, specification in schema.items():
        if name not in values:
            raise ModelTrainingError(f"Enter a value for '{name}'.")
        value = values[name]
        if specification["kind"] == "numeric":
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as error:
                raise ModelTrainingError(f"'{name}' must be a number.") from error
            if not np.isfinite(numeric_value):
                raise ModelTrainingError(f"'{name}' must be a finite number.")
            record[name] = numeric_value
        else:
            text_value = str(value).strip()
            if not text_value:
                raise ModelTrainingError(f"Choose a value for '{name}'.")
            record[name] = text_value
    return pd.DataFrame([record])


def metric_explanations(task: TaskType) -> dict[str, str]:
    """Give plain-English metric definitions for the evaluation screen."""
    if task == "classification":
        return {
            "Accuracy": "The share of test predictions that matched the actual class.",
            "Precision (weighted)": "When the model predicted a class, how often it was right, averaged with class size considered.",
            "Recall (weighted)": "How much of each actual class the model found, averaged with class size considered.",
            "F1 score (weighted)": "A balance of precision and recall; it is useful when classes are uneven.",
        }
    return {
        "MAE": "The average absolute distance between predicted and actual values, in the target's units.",
        "RMSE": "Like MAE but penalises larger errors more strongly, in the target's units.",
        "R²": "How much variation in the test target is explained relative to a simple mean baseline. It can be negative.",
    }


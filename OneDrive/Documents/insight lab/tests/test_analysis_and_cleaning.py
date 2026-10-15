"""Tests for data profiling, transparent insights, and non-destructive cleaning."""

import pandas as pd

from src.analysis import build_profile, generate_data_insights, high_correlation_pairs
from src.cleaning import CleaningOptions, apply_cleaning


def test_profile_reports_quality_signals() -> None:
    frame = pd.DataFrame(
        {
            "score": [1.0, 2.0, None, 1.0],
            "label": ["a", "b", "b", "a"],
            "constant": ["same", "same", "same", "same"],
        }
    )
    profile = build_profile(frame)

    assert profile.rows == 4
    assert profile.columns == 3
    assert profile.missing_cells == 1
    assert profile.duplicate_rows == 1
    assert "score" in profile.numeric_columns
    assert "label" in profile.categorical_columns
    assert profile.column_quality.loc[profile.column_quality["Column"] == "score", "Missing"].item() == 1


def test_rule_based_insights_describe_missing_duplicates_and_constants() -> None:
    frame = pd.DataFrame({"value": [1, 1, None], "constant": ["x", "x", "x"]})
    insights = generate_data_insights(frame)
    joined = " ".join(insights).lower()

    assert "missing" in joined
    assert "duplicate" in joined
    assert "only one observed value" in joined


def test_high_correlation_pairs_returns_related_numeric_columns() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "text": list("abcd")})
    pairs = high_correlation_pairs(frame, threshold=0.9)

    assert pairs == [("a", "b", 1.0)]


def test_cleaning_creates_copy_and_fills_missing_values() -> None:
    source = pd.DataFrame({"age": [10.0, None, 10.0], "city": ["A", None, "A"], "drop_me": [1, 2, 1]})
    options = CleaningOptions(
        selected_columns=["age", "city"],
        remove_duplicates=True,
        missing_strategy="Fill numeric with median; labels with mode",
    )
    cleaned, report = apply_cleaning(source, options)

    assert list(cleaned.columns) == ["age", "city"]
    assert cleaned.isna().sum().sum() == 0
    assert report["duplicates_removed"] == 1
    assert source.isna().sum().sum() == 2  # Source was not changed.


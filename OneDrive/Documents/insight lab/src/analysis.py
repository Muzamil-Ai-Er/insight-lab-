"""Transparent dataset profiling and rule-based insight generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetProfile:
    """A compact, display-friendly profile of a DataFrame."""

    rows: int
    columns: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    missing_cells: int
    duplicate_rows: int
    memory_bytes: int
    column_quality: pd.DataFrame
    summary: pd.DataFrame


def get_column_groups(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical-like column names in original order."""
    numeric = frame.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in frame.columns if column not in numeric]
    return numeric, categorical


def build_profile(frame: pd.DataFrame) -> DatasetProfile:
    """Compute profile statistics without changing the supplied DataFrame."""
    numeric, categorical = get_column_groups(frame)
    missing_by_column = frame.isna().sum()
    non_null = frame.notna().sum()
    examples = [
        "—" if value is None or pd.isna(value) else str(value)[:60]
        for value in (frame[column].dropna().iloc[0] if frame[column].notna().any() else None for column in frame.columns)
    ]
    denominator = max(len(frame), 1)
    column_quality = pd.DataFrame(
        {
            "Column": frame.columns,
            "Data type": frame.dtypes.astype(str).values,
            "Non-null": non_null.values,
            "Missing": missing_by_column.values,
            "Missing %": (missing_by_column / denominator * 100).round(2).values,
            "Unique values": frame.nunique(dropna=True).values,
            "Example value": examples,
        }
    )

    try:
        summary = frame.describe(include="all").transpose().reset_index(names="Column")
    except (ValueError, TypeError):
        summary = pd.DataFrame({"Column": frame.columns})

    return DatasetProfile(
        rows=len(frame),
        columns=len(frame.columns),
        numeric_columns=numeric,
        categorical_columns=categorical,
        missing_cells=int(missing_by_column.sum()),
        duplicate_rows=int(frame.duplicated().sum()),
        memory_bytes=int(frame.memory_usage(deep=True).sum()),
        column_quality=column_quality,
        summary=summary,
    )


def high_correlation_pairs(frame: pd.DataFrame, threshold: float = 0.80) -> list[tuple[str, str, float]]:
    """Find numeric pairs with an absolute Pearson correlation over a threshold."""
    numeric, _ = get_column_groups(frame)
    if len(numeric) < 2:
        return []

    correlation = frame[numeric].corr(numeric_only=True)
    pairs: list[tuple[str, str, float]] = []
    for index, first in enumerate(numeric):
        for second in numeric[index + 1 :]:
            value = correlation.loc[first, second]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append((first, second, float(value)))
    return sorted(pairs, key=lambda item: abs(item[2]), reverse=True)


def generate_data_insights(frame: pd.DataFrame, profile: DatasetProfile | None = None) -> list[str]:
    """Produce plainly-labelled, rule-based observations about a dataset.

    These are deterministic data checks, not output from a generative AI model.
    """
    profile = profile or build_profile(frame)
    insights: list[str] = [
        f"This working dataset contains {profile.rows:,} rows and {profile.columns:,} columns."
    ]
    if profile.rows == 0:
        return insights + ["There are no rows to analyse yet."]

    missing_columns = profile.column_quality.query("Missing > 0").sort_values("Missing", ascending=False)
    if missing_columns.empty:
        insights.append("No missing values were detected.")
    else:
        top = missing_columns.iloc[0]
        insights.append(
            f"{len(missing_columns)} column(s) contain missing values; "
            f"'{top['Column']}' has the most ({int(top['Missing']):,}, {top['Missing %']:.1f}%)."
        )

    if profile.duplicate_rows:
        fraction = profile.duplicate_rows / profile.rows * 100
        insights.append(
            f"{profile.duplicate_rows:,} complete duplicate row(s) were found ({fraction:.1f}% of rows). "
            "Consider removing them only if repeated records are not meaningful."
        )
    else:
        insights.append("No complete duplicate rows were found.")

    constant_columns = profile.column_quality.query("`Unique values` <= 1")
    if not constant_columns.empty:
        names = ", ".join(f"'{name}'" for name in constant_columns["Column"].head(3))
        suffix = "" if len(constant_columns) <= 3 else " (and others)"
        insights.append(f"{names}{suffix} has only one observed value and may add little predictive value.")

    correlations = high_correlation_pairs(frame)
    if correlations:
        first, second, value = correlations[0]
        insights.append(
            f"'{first}' and '{second}' have a strong Pearson correlation ({value:.2f}). "
            "Correlation describes association, not causation."
        )

    if profile.categorical_columns:
        cardinality = profile.column_quality.set_index("Column").loc[profile.categorical_columns, "Unique values"]
        high_cardinality = cardinality[cardinality > max(20, profile.rows * 0.25)]
        if not high_cardinality.empty:
            names = ", ".join(f"'{name}'" for name in high_cardinality.index[:3])
            insights.append(
                f"{names} has many distinct labels. High-cardinality categories can make small models less stable."
            )
    return insights


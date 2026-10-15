"""Non-destructive data-cleaning operations used by the Streamlit interface."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CleaningOptions:
    """User-selected cleaning choices applied to a fresh copy of the data."""

    selected_columns: list[str]
    remove_duplicates: bool = False
    missing_strategy: str = "Keep missing values"
    constant_value: str = "Unknown"


def apply_cleaning(frame: pd.DataFrame, options: CleaningOptions) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return a cleaned copy and a concise change report.

    The function never mutates ``frame``. Invalid selected columns are ignored so
    a UI refresh cannot cause a destructive error.
    """
    selected = [column for column in options.selected_columns if column in frame.columns]
    if not selected:
        raise ValueError("Select at least one column to keep.")

    cleaned = frame.loc[:, selected].copy(deep=True)
    rows_before = len(cleaned)
    missing_before = int(cleaned.isna().sum().sum())
    duplicates_removed = 0

    if options.remove_duplicates:
        duplicates_removed = int(cleaned.duplicated().sum())
        cleaned = cleaned.drop_duplicates().copy()

    strategy = options.missing_strategy
    if strategy == "Drop rows with missing values":
        cleaned = cleaned.dropna().copy()
    elif strategy in {"Fill numeric with median; labels with mode", "Fill numeric with mean; labels with mode"}:
        numeric_columns = cleaned.select_dtypes(include="number").columns
        for column in numeric_columns:
            statistic = cleaned[column].median() if "median" in strategy else cleaned[column].mean()
            if pd.isna(statistic):
                statistic = 0
            cleaned[column] = cleaned[column].fillna(statistic)
        for column in cleaned.columns.difference(numeric_columns):
            non_null = cleaned[column].dropna()
            replacement = non_null.mode().iloc[0] if not non_null.mode().empty else "Unknown"
            cleaned[column] = cleaned[column].fillna(replacement)
    elif strategy == "Fill all missing values with a custom value":
        cleaned = cleaned.fillna(options.constant_value)
    elif strategy != "Keep missing values":
        raise ValueError("Choose a supported missing-value strategy.")

    report = {
        "rows_before": rows_before,
        "rows_after": len(cleaned),
        "duplicates_removed": duplicates_removed,
        "missing_before": missing_before,
        "missing_after": int(cleaned.isna().sum().sum()),
        "columns_kept": len(cleaned.columns),
    }
    return cleaned, report


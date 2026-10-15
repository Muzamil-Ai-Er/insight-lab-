"""Tests for in-memory CSV input and output helpers."""

import pytest

from src.data_io import DatasetLoadError, dataframe_to_csv_bytes, load_csv_bytes


def test_load_csv_bytes_reads_utf8_data() -> None:
    frame, encoding = load_csv_bytes(b"name,value\nAda,10\n", "example.csv")

    assert encoding in {"utf-8-sig", "utf-8"}
    assert frame.to_dict("records") == [{"name": "Ada", "value": 10}]
    assert dataframe_to_csv_bytes(frame).startswith(b"name,value")


def test_load_csv_bytes_rejects_empty_input() -> None:
    with pytest.raises(DatasetLoadError, match="empty"):
        load_csv_bytes(b"", "empty.csv")


"""Central configuration values for the InsightLab application."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA_PATH = PROJECT_ROOT / "sample_data" / "passenger_outcomes.csv"

# The app processes files in memory. This ceiling keeps accidental multi-gigabyte
# uploads from freezing a local browser or a small Community Cloud instance.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_PREVIEW_ROWS = 100
MAX_CHART_ROWS = 5_000
RANDOM_STATE = 42


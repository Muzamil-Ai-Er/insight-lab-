"""Safe, in-memory CSV loading and download helpers."""

from __future__ import annotations

from io import BytesIO

import pandas as pd

from .config import MAX_UPLOAD_BYTES


class DatasetLoadError(ValueError):
    """Raised when a supplied CSV cannot be safely read into a DataFrame."""


def load_csv_bytes(raw_bytes: bytes, file_name: str = "dataset.csv") -> tuple[pd.DataFrame, str]:
    """Load a CSV from bytes without writing it to disk.

    UTF-8 is preferred, with Latin-1 as a pragmatic fallback for common exports
    from spreadsheet applications. Only tabular CSV parsing is performed; no
    uploaded code, formulas, or macros are executed.
    """
    if not raw_bytes:
        raise DatasetLoadError("The selected file is empty.")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise DatasetLoadError(
            f"{file_name} is larger than the {limit_mb} MB in-memory upload limit. "
            "Export a smaller CSV or remove unneeded columns before uploading."
        )

    parse_errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            frame = pd.read_csv(BytesIO(raw_bytes), encoding=encoding, low_memory=False)
            if frame.empty and len(frame.columns) == 0:
                raise DatasetLoadError("The CSV has no readable columns.")
            return frame, encoding
        except UnicodeDecodeError as error:
            parse_errors.append(f"{encoding}: {error}")
        except pd.errors.EmptyDataError as error:
            raise DatasetLoadError("The CSV has no rows or columns.") from error
        except pd.errors.ParserError as error:
            raise DatasetLoadError(
                "The file could not be parsed as CSV. Check its delimiter, quoting, and row structure."
            ) from error
        except DatasetLoadError:
            raise

    raise DatasetLoadError(
        "The file could not be decoded as UTF-8 or Latin-1 CSV. "
        "Save it again as a CSV encoded in UTF-8 and try again."
    )


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Create a UTF-8 CSV download without persisting the user's data."""
    return frame.to_csv(index=False).encode("utf-8")


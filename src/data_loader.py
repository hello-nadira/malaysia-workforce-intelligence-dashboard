"""
Data loading, validation, and caching for the LFS Monthly dataset.

This module is responsible for:
  1. Downloading the CSV from the official DOSM URL (with retries).
  2. Caching it locally as Parquet so the app works even if the download
     fails on a subsequent run.
  3. Validating the schema, date column, duplicates, and numeric types.
  4. Providing a single entry point — load_dataframe() — that the
     Streamlit app and all downstream modules call.

Beginner note: this file contains NO Streamlit imports. It is pure
pandas + requests, so it can be tested with pytest without launching
a web server.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from src import config


# ---------------------------------------------------------------------------
# Custom exceptions — these let the caller distinguish *why* something failed
# ---------------------------------------------------------------------------

class DataDownloadError(Exception):
    """Raised when the CSV cannot be fetched from the source URL."""


class DataValidationError(Exception):
    """Raised when the downloaded DataFrame fails structural validation."""


# ---------------------------------------------------------------------------
# Cache metadata — stored alongside the data to track when the cache
# was created vs. the latest observation date in the data itself.
# ---------------------------------------------------------------------------

@dataclass
class CacheMetadata:
    """Metadata stored with the cached DataFrame."""
    fetch_timestamp: float  # Unix epoch of the successful source fetch
    source_url: str

    def to_dict(self) -> dict:
        return {
            "fetch_timestamp": self.fetch_timestamp,
            "source_url": self.source_url,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CacheMetadata":
        return cls(
            fetch_timestamp=d["fetch_timestamp"],
            source_url=d["source_url"],
        )


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------

def _download_csv_bytes(url: str, max_retries: int = 3, timeout: int = 30) -> bytes:
    """
    Download the CSV file and return its raw bytes.

    Retries up to max_retries times with exponential backoff.
    Raises DataDownloadError if all attempts fail.
    """
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            last_error = exc
            if attempt < max_retries:
                wait = 2 ** attempt  # 2s, 4s, 8s...
                time.sleep(wait)

    raise DataDownloadError(
        f"Failed to download CSV from {url} after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# 2. Cache
# ---------------------------------------------------------------------------

def _cache_dir_exists() -> bool:
    """Check if the cache directory and file both exist."""
    return config.CACHE_FILE.exists() and config.CACHE_FILE.stat().st_size > 0


def _load_from_cache() -> tuple[pd.DataFrame, CacheMetadata]:
    """
    Load the cached DataFrame and metadata from the Parquet file.

    Returns (DataFrame, CacheMetadata).
    Raises FileNotFoundError if cache does not exist.
    """
    if not _cache_dir_exists():
        raise FileNotFoundError("No cached data file found.")

    df = pd.read_parquet(config.CACHE_FILE)

    # Metadata is stored in the DataFrame's attrs (PyArrow-backed Parquet
    # supports pandas attrs). If attrs are missing, construct minimal metadata.
    fetch_ts = df.attrs.get("fetch_timestamp", 0.0)
    source_url = df.attrs.get("source_url", config.CSV_URL)
    metadata = CacheMetadata(fetch_timestamp=fetch_ts, source_url=source_url)

    return df, metadata


def _save_to_cache(df: pd.DataFrame, metadata: CacheMetadata) -> None:
    """
    Save the DataFrame to the Parquet cache, including metadata in attrs.

    Creates the cache directory if it does not exist.
    """
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Attach metadata to the DataFrame's attrs before saving.
    df.attrs["fetch_timestamp"] = metadata.fetch_timestamp
    df.attrs["source_url"] = metadata.source_url

    df.to_parquet(config.CACHE_FILE, index=False)


def get_cache_age_seconds() -> float | None:
    """
    Return the age of the cache in seconds, or None if no cache exists.

    The age is measured from the last *successful source fetch* timestamp
    stored in the cache metadata, NOT from when the file was last touched
    on disk. This satisfies requirement #5.
    """
    if not _cache_dir_exists():
        return None

    try:
        _, metadata = _load_from_cache()
        return time.time() - metadata.fetch_timestamp
    except (FileNotFoundError, KeyError):
        return None


def is_cache_expired() -> bool:
    """
    Check whether the cache is older than CACHE_TTL_SECONDS.

    Returns True if expired (or no cache exists), False if fresh.
    """
    age = get_cache_age_seconds()
    if age is None:
        return True
    return age > config.CACHE_TTL_SECONDS


def clear_cache() -> None:
    """Delete the cached Parquet file if it exists."""
    if config.CACHE_FILE.exists():
        config.CACHE_FILE.unlink()


# ---------------------------------------------------------------------------
# 3. Validate
# ---------------------------------------------------------------------------

def validate_dataframe(df: pd.DataFrame) -> list[str]:
    """
    Validate the structural integrity of the LFS DataFrame.

    Checks performed (in order):
      1. All 8 expected columns are present.
      2. The date column is present and can be parsed as datetime.
      3. Date column has no missing values.
      4. No duplicate date values exist.
      5. All numeric columns are numeric (float64 or int64).
      6. No numeric column is entirely NaN.

    Returns a list of warning messages (e.g. "Column X has N NaN values").
    Raises DataValidationError if a *hard* failure is detected
    (missing columns, duplicate dates, non-numeric columns).

    Missing values within a column are reported as warnings, not errors —
    the dashboard will show "N/A" for those cells rather than silently
    filling or discarding them (requirement #7).
    """
    warnings: list[str] = []

    # --- Check 1: required columns present ---
    missing_cols = [c for c in config.ALL_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(
            f"Missing required columns: {missing_cols}. "
            f"Expected: {config.ALL_COLUMNS}"
        )

    # --- Check 2: date column is parsable ---
    if not pd.api.types.is_datetime64_any_dtype(df[config.DATE_COLUMN]):
        try:
            df[config.DATE_COLUMN] = pd.to_datetime(df[config.DATE_COLUMN])
        except (ValueError, TypeError) as exc:
            raise DataValidationError(
                f"Date column '{config.DATE_COLUMN}' cannot be parsed as datetime: {exc}"
            )

    # --- Check 3: no missing dates ---
    null_dates = df[config.DATE_COLUMN].isna().sum()
    if null_dates > 0:
        raise DataValidationError(
            f"Date column '{config.DATE_COLUMN}' has {null_dates} missing values. "
            "Dates are required for all records."
        )

    # --- Check 4: no duplicate dates ---
    dup_count = df[config.DATE_COLUMN].duplicated().sum()
    if dup_count > 0:
        dup_dates = df.loc[df[config.DATE_COLUMN].duplicated(), config.DATE_COLUMN].tolist()
        raise DataValidationError(
            f"Duplicate date values found: {dup_count} records. "
            f"Duplicate dates: {dup_dates[:10]}"
        )

    # --- Check 5: numeric columns are numeric ---
    for col in config.NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(
                f"Column '{col}' is not numeric (dtype={df[col].dtype}). "
                "Numeric columns must contain float or int values."
            )

    # --- Check 6: no fully-empty numeric column ---
    for col in config.NUMERIC_COLUMNS:
        if df[col].isna().all():
            raise DataValidationError(
                f"Column '{col}' is entirely empty (all NaN)."
            )

    # --- Warnings: report partial missing values ---
    for col in config.ALL_COLUMNS:
        n_null = df[col].isna().sum()
        if n_null > 0:
            pct = (n_null / len(df)) * 100
            warnings.append(
                f"Column '{col}' has {n_null} missing values ({pct:.1f}%)."
            )

    if warnings:
        warnings.insert(0, "Data validation completed with warnings.")
    else:
        warnings.append("Data validation passed with no warnings.")

    return warnings


# ---------------------------------------------------------------------------
# 4. Parse / clean
# ---------------------------------------------------------------------------

def _parse_and_clean(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw DataFrame into the canonical form:
      - Parse date column to datetime
      - Ensure numeric columns are float64
      - Sort by date ascending
      - Reset index

    This does NOT fill or drop missing values — those are reported
    by validate_dataframe and handled downstream.
    """
    df = raw_df.copy()

    # Parse dates
    df[config.DATE_COLUMN] = pd.to_datetime(df[config.DATE_COLUMN])

    # Coerce numeric columns (handles any string values that should be numbers)
    for col in config.NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort chronologically
    df = df.sort_values(config.DATE_COLUMN, ascending=True).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# 5. Public API
# ---------------------------------------------------------------------------

def load_dataframe(
    force_refresh: bool = False,
    allow_cache_fallback: bool = True,
) -> tuple[pd.DataFrame, "CacheMetadata", list[str]]:
    """
    Load the LFS dataset with caching and validation.

    Parameters
    ----------
    force_refresh : bool
        If True, ignore the cache and download fresh data from the source.
    allow_cache_fallback : bool
        If True and the download fails, fall back to the cached version.
        If False and the download fails, re-raise DataDownloadError.

    Returns
    -------
    tuple of (DataFrame, CacheMetadata, list[str])
        The validated DataFrame, cache metadata, and a list of
        human-readable status/warning messages.

    Raises
    ------
    DataDownloadError
        If download fails and no cache is available (or fallback disabled).
    DataValidationError
        If the downloaded or cached data fails validation.

    Beginner note: This function is the "front door" for all data access.
    Both the Streamlit app and the test suite call it — everything else
    in the codebase depends on the DataFrame it returns.
    """
    messages: list[str] = []
    metadata: CacheMetadata | None = None

    # --- Try to use cache if not forced to refresh ---
    if not force_refresh and _cache_dir_exists():
        df, metadata = _load_from_cache()
        messages.append(
            f"Loaded data from cache (last successful fetch: "
            f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.localtime(metadata.fetch_timestamp))})."
        )
        df = _parse_and_clean(df)
        _ = validate_dataframe(df)  # validate cached data
        messages.append("Cached data validated successfully.")
        return df, metadata, messages

    # --- Download fresh data ---
    try:
        raw_bytes = _download_csv_bytes(config.CSV_URL)
        df = pd.read_csv(pd.io.common.BytesIO(raw_bytes))
        df = _parse_and_clean(df)
        validation_warnings = validate_dataframe(df)

        metadata = CacheMetadata(
            fetch_timestamp=time.time(),
            source_url=config.CSV_URL,
        )
        _save_to_cache(df, metadata)
        messages.append(f"Downloaded and cached {len(df)} records from {config.CSV_URL}.")

        if validation_warnings:
            messages.extend(validation_warnings)

        return df, metadata, messages

    except DataDownloadError as exc:
        messages.append(f"Download failed: {exc}")

        if allow_cache_fallback and _cache_dir_exists():
            df, metadata = _load_from_cache()
            df = _parse_and_clean(df)
            _ = validate_dataframe(df)
            messages.append(
                "Falling back to cached data. "
                "Last successful source fetch: "
                f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.localtime(metadata.fetch_timestamp))}."
            )
            messages.append("WARNING: Showing cached data (not live).")
            return df, metadata, messages

        # No cache available
        messages.append(
            "No cached data available as fallback. "
            "Please check your internet connection and try again."
        )
        raise


# ---------------------------------------------------------------------------
# Helpers for metadata
# ---------------------------------------------------------------------------

def get_latest_observation_date(df: pd.DataFrame) -> pd.Timestamp:
    """Return the most recent date in the DataFrame."""
    return df[config.DATE_COLUMN].max()


def get_data_range(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (earliest_date, latest_date) in the DataFrame."""
    return (df[config.DATE_COLUMN].min(), df[config.DATE_COLUMN].max())



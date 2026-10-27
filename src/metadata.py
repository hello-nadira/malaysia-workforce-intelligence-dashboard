"""
Metadata helpers for data freshness and source attribution.

This module provides:
  - get_last_fetch_timestamp() — when we last successfully downloaded data
  - get_latest_observation_date() — the most recent month in the data
  - get_source_attribution() — the full source/license citation
  - format_fetch_timestamp() — human-readable last-fetch time

These helpers keep the Streamlit UI focused on presentation while all
data-source knowledge lives here.

This module contains NO Streamlit imports.
"""

from __future__ import annotations

import time
from datetime import timezone

import pandas as pd

from src import config
from src.data_loader import CacheMetadata, get_cache_age_seconds


def get_last_fetch_timestamp() -> pd.Timestamp | None:
    """
    Return the timestamp of the last successful source download.

    This comes from the cache metadata, NOT from the data itself.
    If no cache exists, returns None.

    This satisfies requirement #5: the cache timestamp represents the
    last successful *source fetch*, separately from the latest *observation
    date*.
    """
    age = get_cache_age_seconds()
    if age is None:
        return None
    # fetch_timestamp = now - age
    fetch_epoch = time.time() - age
    return pd.Timestamp(fetch_epoch, unit="s")


def get_latest_observation_date(df: pd.DataFrame) -> pd.Timestamp:
    """
    Return the most recent observation date in the DataFrame.

    This is the latest month in the data (e.g. 2026-08-01), NOT the
    fetch timestamp.
    """
    return df[config.DATE_COLUMN].max()


def get_source_attribution() -> str:
    """
    Return the full source attribution string for display.

    Example:
        "Source: Department of Statistics Malaysia (DOSM) | "
        "License: Creative Commons Attribution 4.0 International (CC BY 4.0)"
    """
    return (
        f"Source: {config.SOURCE_NAME} | "
        f"License: {config.LICENSE}"
    )


def get_source_url() -> str:
    """Return the direct CSV download URL."""
    return config.CSV_URL


def format_fetch_timestamp(fetch_ts: pd.Timestamp) -> str:
    """
    Format a fetch timestamp for human-readable display.

    Includes UTC timezone for clarity.
    """
    # Convert to UTC for display
    return fetch_ts.strftime("%Y-%m-%d %H:%M:%S UTC")


def get_data_summary_text(
    df: pd.DataFrame,
    metadata: CacheMetadata,
) -> str:
    """
    Produce a single summary string combining:
      - latest observation date
      - last successful fetch timestamp
      - number of records
      - cache age

    Used in the Streamlit sidebar or footer.
    """
    latest_obs = get_latest_observation_date(df)
    fetch_ts = pd.Timestamp(metadata.fetch_timestamp, unit="s")
    age_seconds = time.time() - metadata.fetch_timestamp
    age_minutes = age_seconds / 60

    parts = [
        f"Data as of {latest_obs.strftime('%B %Y')}",
        f"Last fetched: {format_fetch_timestamp(fetch_ts)} ({age_minutes:.0f} min ago)",
        f"Records: {len(df)}",
    ]

    return " | ".join(parts)

"""
Tests for date comparisons and date-range filtering in src/kpi.py.

Tests cover:
  - filter_by_date_range() with various start/end combinations
  - crosses_trend_break() for various date pairs
  - get_available_date_range() helper
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.kpi import (
    crosses_trend_break,
    filter_by_date_range,
    get_available_date_range,
)


# ---------------------------------------------------------------------------
# Helper: build a test DataFrame spanning pre-2025 to post-2025
# ---------------------------------------------------------------------------

def _make_test_df_dates() -> pd.DataFrame:
    """
    Build a DataFrame with dates spanning the 2025 trend-break boundary.

    Rows: Jan 2024, Jun 2024, Dec 2024, Jan 2025, Jun 2025, Dec 2025, Jun 2026
    """
    dates = pd.to_datetime([
        "2024-01-01", "2024-06-01", "2024-12-01",
        "2025-01-01", "2025-06-01", "2025-12-01",
        "2026-06-01",
    ])
    data = {
        config.DATE_COLUMN: dates,
        "lf": [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0],
        "lf_employed": [95.0, 97.0, 99.0, 101.0, 103.0, 105.0, 107.0],
        "lf_unemployed": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        "lf_outside": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
        "p_rate": [64.0, 65.0, 64.5, 65.5, 66.0, 66.5, 67.0],
        "ep_ratio": [61.0, 62.0, 61.5, 62.5, 63.0, 63.5, 64.0],
        "u_rate": [3.5, 3.0, 3.2, 4.8, 3.5, 3.2, 3.0],  # higher in Jan 2025 due to pop-ref change
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# filter_by_date_range
# ---------------------------------------------------------------------------

class TestFilterByDateRange:
    def test_filter_start_only(self):
        df = _make_test_df_dates()
        result = filter_by_date_range(df, "2025-01-01", None)
        assert len(result) == 4  # Jan 2025, Jun 2025, Dec 2025, Jun 2026
        assert result.iloc[0][config.DATE_COLUMN] == pd.Timestamp("2025-01-01")

    def test_filter_end_only(self):
        df = _make_test_df_dates()
        result = filter_by_date_range(df, None, "2024-06-01")
        assert len(result) == 2  # Jan 2024 and Jun 2024
        assert result.iloc[-1][config.DATE_COLUMN] == pd.Timestamp("2024-06-01")

    def test_filter_both_bounds(self):
        df = _make_test_df_dates()
        result = filter_by_date_range(df, "2024-06-01", "2025-06-01")
        assert len(result) == 4  # Jun 2024, Dec 2024, Jan 2025, Jun 2025
        assert result.iloc[0][config.DATE_COLUMN] == pd.Timestamp("2024-06-01")
        assert result.iloc[-1][config.DATE_COLUMN] == pd.Timestamp("2025-06-01")

    def test_filter_no_bounds_returns_all(self):
        df = _make_test_df_dates()
        result = filter_by_date_range(df, None, None)
        assert len(result) == len(df)

    def test_filter_outside_range_returns_empty(self):
        df = _make_test_df_dates()
        result = filter_by_date_range(df, "2023-01-01", "2023-12-01")
        assert len(result) == 0

    def test_filter_string_date_input(self):
        df = _make_test_df_dates()
        result = filter_by_date_range(df, "2025-01-01", "2026-06-01")
        assert len(result) == 4

    def test_filter_returns_sorted(self):
        df = _make_test_df_dates()
        # Shuffle the DataFrame to ensure filter_by_date_range sorts it
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        result = filter_by_date_range(df_shuffled, "2024-01-01", "2026-06-01")
        assert result[config.DATE_COLUMN].is_monotonic_increasing

    def test_filter_inclusive_boundaries(self):
        df = _make_test_df_dates()
        result = filter_by_date_range(df, "2024-12-01", "2024-12-01")
        assert len(result) == 1
        assert result.iloc[0][config.DATE_COLUMN] == pd.Timestamp("2024-12-01")


# ---------------------------------------------------------------------------
# get_available_date_range
# ---------------------------------------------------------------------------

class TestGetAvailableDateRange:
    def test_returns_min_and_max(self):
        df = _make_test_df_dates()
        start, end = get_available_date_range(df)
        assert start == pd.Timestamp("2024-01-01")
        assert end == pd.Timestamp("2026-06-01")

    def test_single_row(self):
        df = _make_test_df_dates().head(1)
        start, end = get_available_date_range(df)
        assert start == pd.Timestamp("2024-01-01")
        assert end == pd.Timestamp("2024-01-01")


# ---------------------------------------------------------------------------
# crosses_trend_break edge cases
# ---------------------------------------------------------------------------

class TestCrossesTrendBreakDates:
    def test_dec2024_to_jan2025_crosses(self):
        """Dec 2024 → Jan 2025 crosses the break."""
        result = crosses_trend_break(
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2024-12-01"),
        )
        assert result is True

    def test_jan2025_to_feb2025_no_cross(self):
        """Both after break — no crossing."""
        result = crosses_trend_break(
            pd.Timestamp("2025-02-01"),
            pd.Timestamp("2025-01-01"),
        )
        assert result is False

    def test_jun2024_to_dec2024_no_cross(self):
        """Both before break — no crossing."""
        result = crosses_trend_break(
            pd.Timestamp("2024-12-01"),
            pd.Timestamp("2024-06-01"),
        )
        assert result is False

    def test_long_span_crosses(self):
        """Jan 2026 → Jan 2024 crosses the break."""
        result = crosses_trend_break(
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2024-01-01"),
        )
        assert result is True

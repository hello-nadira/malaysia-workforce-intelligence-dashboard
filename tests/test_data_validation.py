"""
Tests for data validation in src/data_loader.py.

These tests use small hand-crafted DataFrames with known defects so the
results are deterministic and do not depend on the live CSV or internet.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.data_loader import (
    CacheMetadata,
    DataValidationError,
    _download_csv_bytes,
    _parse_and_clean,
    validate_dataframe,
)


# ---------------------------------------------------------------------------
# Helper: build a valid baseline DataFrame
# ---------------------------------------------------------------------------

def _make_valid_df(n_rows: int = 3) -> pd.DataFrame:
    """Return a small DataFrame that passes all validation checks."""
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="MS")
    data = {
        config.DATE_COLUMN: dates,
        "lf": [100.0 + i for i in range(n_rows)],
        "lf_employed": [95.0 + i for i in range(n_rows)],
        "lf_unemployed": [5.0 + i for i in range(n_rows)],
        "lf_outside": [50.0 + i for i in range(n_rows)],
        "p_rate": [64.0 + i for i in range(n_rows)],
        "ep_ratio": [61.0 + i for i in range(n_rows)],
        "u_rate": [3.5 + i for i in range(n_rows)],
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Check 1: All required columns present
# ---------------------------------------------------------------------------

class TestMissingColumns:
    def test_missing_one_column_raises(self):
        df = _make_valid_df()
        df = df.drop(columns=["u_rate"])
        with pytest.raises(DataValidationError, match="Missing required columns"):
            validate_dataframe(df)

    def test_missing_multiple_columns_raises(self):
        df = _make_valid_df()
        df = df.drop(columns=["lf", "u_rate", "p_rate"])
        with pytest.raises(DataValidationError, match="Missing required columns"):
            validate_dataframe(df)



# ---------------------------------------------------------------------------
# Check 2: Date column is parsable as datetime
# ---------------------------------------------------------------------------

class TestDateParsing:
    def test_unparseable_date_raises(self):
        """Setting an unparseable string into a datetime column should fail."""
        raw = pd.DataFrame({
            "date": ["2024-01-01", "not-a-date"],
            "lf": [100.0, 101.0],
            "lf_employed": [95.0, 96.0],
            "lf_unemployed": [5.0, 5.0],
            "lf_outside": [50.0, 51.0],
            "p_rate": [64.0, 64.5],
            "ep_ratio": [61.0, 61.5],
            "u_rate": [3.5, 3.4],
        })
        with pytest.raises(DataValidationError, match="cannot be parsed as datetime"):
            validate_dataframe(raw)

    def test_string_dates_are_parsed(self):
        df = _make_valid_df()
        # Replace the entire column with strings to simulate raw CSV import
        df[config.DATE_COLUMN] = df[config.DATE_COLUMN].dt.strftime("%Y-%m-%d")
        warnings = validate_dataframe(df)
        assert pd.api.types.is_datetime64_any_dtype(df[config.DATE_COLUMN])
        assert any("no warnings" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# Check 3: No missing dates
# ---------------------------------------------------------------------------

class TestMissingDates:
    def test_missing_date_value_raises(self):
        df = _make_valid_df()
        df.loc[1, config.DATE_COLUMN] = pd.NaT
        with pytest.raises(DataValidationError, match="missing values"):
            validate_dataframe(df)


# ---------------------------------------------------------------------------
# Check 4: No duplicate dates
# ---------------------------------------------------------------------------

class TestDuplicateDates:
    def test_duplicate_dates_raise(self):
        df = _make_valid_df(n_rows=3)
        df.loc[2, config.DATE_COLUMN] = df.loc[0, config.DATE_COLUMN]
        with pytest.raises(DataValidationError, match="Duplicate date values"):
            validate_dataframe(df)

    def test_no_duplicates_passes(self):
        df = _make_valid_df(n_rows=5)
        warnings = validate_dataframe(df)
        assert any("no warnings" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# Check 5: Numeric columns are numeric
# ---------------------------------------------------------------------------

class TestNumericColumns:
    def test_string_in_numeric_column_raises(self):
        """A non-numeric string in a numeric column should fail validation."""
        raw = pd.DataFrame({
            "date": ["2024-01-01", "2024-02-01"],
            "lf": [100.0, 101.0],
            "lf_employed": [95.0, 96.0],
            "lf_unemployed": [5.0, 5.0],
            "lf_outside": [50.0, 51.0],
            "p_rate": [64.0, 64.5],
            "ep_ratio": [61.0, 61.5],
            "u_rate": ["abc", "3.4"],  # string sneaks in
        })
        with pytest.raises(DataValidationError, match="is not numeric"):
            validate_dataframe(raw)

    def test_float_column_passes(self):
        df = _make_valid_df()
        warnings = validate_dataframe(df)
        assert any("no warnings" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# Check 6: No fully-empty numeric column
# ---------------------------------------------------------------------------

class TestEmptyColumns:
    def test_all_nan_column_raises(self):
        """A column that is entirely NaN should fail validation."""
        raw = pd.DataFrame({
            "date": ["2024-01-01", "2024-02-01"],
            "lf": [100.0, 101.0],
            "lf_employed": [95.0, 96.0],
            "lf_unemployed": [5.0, 5.0],
            "lf_outside": [50.0, 51.0],
            "p_rate": [64.0, 64.5],
            "ep_ratio": [61.0, 61.5],
            "u_rate": [float("nan"), float("nan")],  # all NaN, but float dtype
        })
        with pytest.raises(DataValidationError, match="entirely empty"):
            validate_dataframe(raw)


# ---------------------------------------------------------------------------
# Warnings: partial missing values are reported, not fatal
# ---------------------------------------------------------------------------

class TestPartialMissing:
    def test_one_missing_value_is_warning(self):
        """A single NaN in a numeric column should produce a warning, not an error."""
        df = _make_valid_df(n_rows=5)
        # Use numpy nan to preserve float64 dtype
        import numpy as np
        df.loc[2, "u_rate"] = np.nan
        warnings = validate_dataframe(df)
        assert any("u_rate" in w and "missing" in w.lower() for w in warnings)

    def test_multiple_missing_values_reported(self):
        """Multiple NaNs across different columns should each be reported."""
        import numpy as np
        df = _make_valid_df(n_rows=5)
        df.loc[0, "lf"] = np.nan
        df.loc[1, "p_rate"] = np.nan
        warnings = validate_dataframe(df)
        assert any("lf" in w and "missing" in w.lower() for w in warnings)
        assert any("p_rate" in w and "missing" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# _parse_and_clean
# ---------------------------------------------------------------------------

class TestParseAndClean:
    def test_parses_dates_and_numerics(self):
        raw = pd.DataFrame({
            "date": ["2024-01-01", "2024-02-01"],
            "lf": [100.0, 101.0],
            "lf_employed": [95.0, 96.0],
            "lf_unemployed": [5.0, 5.0],
            "lf_outside": [50.0, 50.0],
            "p_rate": [64.0, 64.5],
            "ep_ratio": [61.0, 61.5],
            "u_rate": [3.5, 3.4],
        })
        df = _parse_and_clean(raw)
        assert pd.api.types.is_datetime64_any_dtype(df[config.DATE_COLUMN])
        assert df[config.DATE_COLUMN].is_monotonic_increasing
        assert df.index[0] == 0  # reset index

    def test_sorts_ascending(self):
        raw = pd.DataFrame({
            "date": ["2024-02-01", "2024-01-01", "2024-03-01"],
            "lf": [101.0, 100.0, 102.0],
            "lf_employed": [96.0, 95.0, 97.0],
            "lf_unemployed": [5.0, 5.0, 5.0],
            "lf_outside": [50.0, 50.0, 50.0],
            "p_rate": [64.5, 64.0, 65.0],
            "ep_ratio": [61.5, 61.0, 62.0],
            "u_rate": [3.4, 3.5, 3.3],
        })
        df = _parse_and_clean(raw)
        assert df[config.DATE_COLUMN].iloc[0] == pd.Timestamp("2024-01-01")
        assert df[config.DATE_COLUMN].iloc[2] == pd.Timestamp("2024-03-01")

    def test_coerces_string_numbers(self):
        raw = pd.DataFrame({
            "date": ["2024-01-01"],
            "lf": ["100.0"],
            "lf_employed": ["95.0"],
            "lf_unemployed": ["5.0"],
            "lf_outside": ["50.0"],
            "p_rate": ["64.0"],
            "ep_ratio": ["61.0"],
            "u_rate": ["3.5"],
        })
        df = _parse_and_clean(raw)
        for col in config.NUMERIC_COLUMNS:
            assert df[col].dtype == "float64"



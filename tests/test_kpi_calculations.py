"""
Tests for KPI calculations in src/kpi.py.

Tests cover:
  - format_kpi() with various units and None
  - get_latest_kpi() extraction
  - compute_mom_change() for both rate and count KPIs
  - crosses_trend_break() detection
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.kpi import (
    KPIChange,
    get_latest_kpi,
    compute_mom_change,
    compute_yoy_change,
    crosses_trend_break,
    format_change,
    format_kpi,
    is_favorable_change,
    get_favorability,
    ChangeDisplay,
)


# ---------------------------------------------------------------------------
# Helper: build a small test DataFrame with known values
# ---------------------------------------------------------------------------

def _make_test_df() -> pd.DataFrame:
    """
    Build a 4-row DataFrame with known values for deterministic testing.

    Dates: Jan 2024, Feb 2024, Jan 2025, Feb 2025
    Values chosen so we can verify percentage-point vs relative % logic.
    """
    dates = pd.to_datetime(["2024-01-01", "2024-02-01", "2025-01-01", "2025-02-01"])
    data = {
        config.DATE_COLUMN: dates,
        "lf": [100.0, 102.0, 105.0, 110.0],  # count KPI
        "lf_employed": [95.0, 97.0, 100.0, 105.0],
        "lf_unemployed": [5.0, 5.0, 5.0, 5.0],
        "lf_outside": [50.0, 50.0, 50.0, 50.0],
        "p_rate": [64.0, 65.0, 66.0, 67.0],  # rate KPI
        "ep_ratio": [61.0, 62.0, 63.0, 64.0],  # rate KPI
        "u_rate": [3.5, 3.0, 4.0, 3.5],  # rate KPI
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# format_kpi
# ---------------------------------------------------------------------------

class TestFormatKPI:
    def test_format_percent(self):
        assert format_kpi(3.5, "%") == "3.5%"

    def test_format_percent_one_decimal(self):
        assert format_kpi(3.55, "%") == "3.5%"  # Python uses banker's rounding (round-half-to-even)
        assert format_kpi(3.56, "%") == "3.6%"
        assert format_kpi(3.65, "%") == "3.6%"  # banker's rounding rounds to even (6)
        assert format_kpi(3.75, "%") == "3.8%"

    def test_format_k_persons(self):
        # 17386.0K persons = 17.4M people
        assert format_kpi(17386.0, "K persons") == "17.4M people"

    def test_format_k_persons_large_number(self):
        # 1234567.8K persons = 1234.6M people
        assert format_kpi(1234567.8, "K persons") == "1234.6M people"

    def test_format_none_value(self):
        assert format_kpi(None, "%") == "N/A"

    def test_format_nan_value(self):
        assert format_kpi(float("nan"), "%") == "N/A"

    def test_format_zero(self):
        assert format_kpi(0.0, "%") == "0.0%"

    def test_format_unknown_unit(self):
        result = format_kpi(42.0, "widgets")
        assert "42.0" in result
        assert "widgets" in result


# ---------------------------------------------------------------------------
# get_latest_kpi
# ---------------------------------------------------------------------------

class TestGetLatestKPI:
    def test_latest_u_rate(self):
        df = _make_test_df()
        result = get_latest_kpi(df, "u_rate")
        assert result.value == 3.5
        assert result.date == pd.Timestamp("2025-02-01")
        assert result.is_missing is False

    def test_latest_lf(self):
        df = _make_test_df()
        result = get_latest_kpi(df, "lf")
        assert result.value == 110.0
        assert result.is_missing is False

    def test_latest_when_none_value(self):
        """If the latest value is NaN, is_missing should be True."""
        df = _make_test_df()
        df.loc[3, "u_rate"] = float("nan")
        result = get_latest_kpi(df, "u_rate")
        assert result.is_missing is True
        assert result.value is None

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=config.ALL_COLUMNS)
        df[config.DATE_COLUMN] = pd.to_datetime(df[config.DATE_COLUMN])
        result = get_latest_kpi(df, "u_rate")
        assert result.is_missing is True
        assert result.value is None

    def test_unknown_kpi_key_raises(self):
        df = _make_test_df()
        with pytest.raises(KeyError, match="Unknown KPI key"):
            get_latest_kpi(df, "nonexistent")


# ---------------------------------------------------------------------------
# compute_mom_change — rate KPI
# ---------------------------------------------------------------------------

class TestMoMChangeRate:
    def test_mom_rate_change_value(self):
        """Feb 2025 u_rate (3.5) vs Jan 2025 u_rate (4.0): -0.5 pp."""
        df = _make_test_df()
        df_recent = df.tail(2)  # Jan 2025, Feb 2025
        change = compute_mom_change(df_recent, "u_rate")
        assert change.current_value == 3.5
        assert change.previous_value == 4.0
        assert change.delta_value == pytest.approx(-0.5)
        assert change.is_missing_comparison is False

    def test_mom_rate_no_cross_when_both_post_2025(self):
        """Jan 2025 vs Feb 2025: both after break, no crossing."""
        df = _make_test_df()
        df_subset = df.iloc[[2, 3]]  # Jan 2025, Feb 2025
        change = compute_mom_change(df_subset, "u_rate")
        assert change.crosses_trend_break is False

    def test_mom_rate_crosses_trend_break(self):
        """Jan 2025 vs Jan 2024: crosses the 2025 break."""
        df = _make_test_df()
        df_subset = df.iloc[[0, 2]]  # Jan 2024, Jan 2025
        change = compute_mom_change(df_subset, "u_rate")
        assert change.crosses_trend_break is True
        assert change.current_date == pd.Timestamp("2025-01-01")
        assert change.previous_date == pd.Timestamp("2024-01-01")


# ---------------------------------------------------------------------------
# compute_mom_change — count KPI
# ---------------------------------------------------------------------------

class TestMoMChangeCount:
    def test_mom_count_increase(self):
        """Feb 2025 lf (110) vs Jan 2025 lf (105): +5 thousand, +4.76%."""
        df = _make_test_df()
        df_recent = df.tail(2)
        change = compute_mom_change(df_recent, "lf")
        assert change.current_value == 110.0
        assert change.previous_value == 105.0
        assert change.delta_value == pytest.approx(5.0)
        assert change.delta_percent == pytest.approx(4.76, rel=0.01)

    def test_mom_count_crosses_trend_break(self):
        df = _make_test_df()
        df_subset = df.iloc[[0, 2]]  # Jan 2024, Jan 2025
        change = compute_mom_change(df_subset, "lf")
        assert change.crosses_trend_break is True


# ---------------------------------------------------------------------------
# compute_mom_change — edge cases
# ---------------------------------------------------------------------------

class TestMoMEdgeCases:
    def test_single_row_returns_missing_comparison(self):
        df = _make_test_df().head(1)
        change = compute_mom_change(df, "u_rate")
        assert change.is_missing_comparison is True
        assert change.previous_value is None

    def test_missing_current_value(self):
        df = _make_test_df()
        df.loc[3, "u_rate"] = float("nan")
        change = compute_mom_change(df, "u_rate")
        assert change.is_missing_comparison is True

    def test_missing_previous_value(self):
        df = _make_test_df()
        df.loc[2, "u_rate"] = float("nan")
        change = compute_mom_change(df, "u_rate")
        assert change.is_missing_comparison is True


# ---------------------------------------------------------------------------
# compute_yoy_change
# ---------------------------------------------------------------------------

class TestYoYChange:
    def test_yoy_rate_feb2025_vs_feb2024(self):
        """Feb 2025 u_rate (3.5) vs Feb 2024 u_rate (3.0): +0.5 pp."""
        df = _make_test_df()
        change = compute_yoy_change(df, "u_rate")
        assert change.current_value == 3.5
        assert change.previous_value == 3.0
        assert change.delta_value == pytest.approx(0.5)
        assert change.previous_date == pd.Timestamp("2024-02-01")

    def test_yoy_count_lf(self):
        """Feb 2025 lf (110) vs Feb 2024 lf (102): +8.0, +7.84%."""
        df = _make_test_df()
        change = compute_yoy_change(df, "lf")
        assert change.current_value == 110.0
        assert change.previous_value == 102.0
        assert change.delta_value == pytest.approx(8.0)
        assert change.delta_percent == pytest.approx(7.84, rel=0.01)

    def test_yoy_crosses_trend_break(self):
        """Feb 2025 (post-break) vs Feb 2024 (pre-break) crosses."""
        df = _make_test_df()
        change = compute_yoy_change(df, "u_rate")
        assert change.crosses_trend_break is True

    def test_yoy_no_cross_when_both_pre_2025(self):
        """If latest date is before 2025, no trend-break crossing."""
        df = _make_test_df().head(2)  # Jan 2024, Feb 2024
        change = compute_yoy_change(df, "u_rate")
        assert change.crosses_trend_break is False
        assert change.is_missing_comparison is True  # no prior-year data


# ---------------------------------------------------------------------------
# crosses_trend_break
# ---------------------------------------------------------------------------

class TestCrossesTrendBreak:
    def test_post_vs_pre_crosses(self):
        result = crosses_trend_break(
            pd.Timestamp("2025-06-01"),
            pd.Timestamp("2024-06-01"),
        )
        assert result is True

    def test_post_vs_post_no_cross(self):
        result = crosses_trend_break(
            pd.Timestamp("2025-06-01"),
            pd.Timestamp("2025-05-01"),
        )
        assert result is False

    def test_pre_vs_pre_no_cross(self):
        result = crosses_trend_break(
            pd.Timestamp("2024-06-01"),
            pd.Timestamp("2024-05-01"),
        )
        assert result is False

    def test_none_dates_return_false(self):
        result = crosses_trend_break(None, pd.Timestamp("2024-06-01"))
        assert result is False
        result = crosses_trend_break(pd.Timestamp("2025-06-01"), None)
        assert result is False

    def test_exact_break_date_current(self):
        """Current date = Jan 2025 (the break date) counts as post-break."""
        result = crosses_trend_break(
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2024-12-01"),
        )
        assert result is True


# ---------------------------------------------------------------------------
# format_change — rate KPI
# ---------------------------------------------------------------------------

class TestFormatChangeRate:
    """Tests for format_change() with rate KPIs (percentage-point primary)."""

    def test_rate_kpi_pp_is_primary(self):
        """Rate KPI should show percentage points as the primary metric."""
        change = KPIChange(
            current_value=3.5,
            previous_value=3.0,
            delta_value=0.5,
            delta_percent=16.67,
            current_date=pd.Timestamp("2025-02-01"),
            previous_date=pd.Timestamp("2025-01-01"),
            crosses_trend_break=False,
            is_missing_comparison=False,
            comparison_type="MoM",
        )
        result = format_change("u_rate", change)
        assert isinstance(result, ChangeDisplay)
        assert "0.5" in result.delta_str
        assert "pp" in result.delta_str
        # current 3.5 vs previous 3.0 means rate went UP
        assert "\u2191" in result.delta_str

    def test_rate_kpi_shows_up_arrow_for_increase(self):
        """Rate increase (higher rate) shows up arrow."""
        change = KPIChange(
            current_value=3.5,
            previous_value=3.0,
            delta_value=0.5,
            delta_percent=16.67,
            current_date=pd.Timestamp("2025-02-01"),
            previous_date=pd.Timestamp("2025-01-01"),
            crosses_trend_break=False,
            is_missing_comparison=False,
            comparison_type="MoM",
        )
        result = format_change("u_rate", change)
        assert result.delta_str.startswith("\u2191")
        assert result.direction == "up"

    def test_rate_kpi_shows_down_arrow_for_decrease(self):
        """Rate decrease shows down arrow."""
        change = KPIChange(
            current_value=2.5,
            previous_value=3.0,
            delta_value=-0.5,
            delta_percent=-16.67,
            current_date=pd.Timestamp("2025-02-01"),
            previous_date=pd.Timestamp("2025-01-01"),
            crosses_trend_break=False,
            is_missing_comparison=False,
            comparison_type="MoM",
        )
        result = format_change("u_rate", change)
        assert result.delta_str.startswith("\u2193")
        assert result.direction == "down"

    def test_rate_kpi_no_contradictory_arrows(self):
        """There must be exactly one arrow — no up-down combination."""
        change = KPIChange(
            current_value=3.5,
            previous_value=3.0,
            delta_value=0.5,
            delta_percent=16.67,
            current_date=pd.Timestamp("2025-02-01"),
            previous_date=pd.Timestamp("2025-01-01"),
            crosses_trend_break=False,
            is_missing_comparison=False,
            comparison_type="MoM",
        )
        result = format_change("u_rate", change)
        arrow_count = result.delta_str.count("\u2191") + result.delta_str.count("\u2193")
        assert arrow_count == 1, f"Expected 1 arrow, found {arrow_count}"

    def test_rate_kpi_context_shows_rel_label(self):
        """Rate KPIs should show relative % labeled as 'rel:'."""
        change = KPIChange(
            current_value=3.5,
            previous_value=3.0,
            delta_value=0.5,
            delta_percent=16.67,
            current_date=pd.Timestamp("2025-02-01"),
            previous_date=pd.Timestamp("2025-01-01"),
            crosses_trend_break=False,
            is_missing_comparison=False,
            comparison_type="MoM",
        )
        result = format_change("u_rate", change)
        assert "rel:" in result.context_str
        assert "16.7" in result.context_str

    def test_rate_kpi_no_context_when_no_relative_change(self):
        """If delta_percent is None, context_str should be empty."""
        change = KPIChange(
            current_value=3.5,
            previous_value=3.0,
            delta_value=0.5,
            delta_percent=None,
            current_date=pd.Timestamp("2025-02-01"),
            previous_date=pd.Timestamp("2025-01-01"),
            crosses_trend_break=False,
            is_missing_comparison=False,
            comparison_type="MoM",
        )
        result = format_change("u_rate", change)
        assert result.context_str == ""

    def test_rate_kpi_break_flag_present(self):
        """Trend-break flag should be in break_flag."""
        change = KPIChange(
            current_value=4.0,
            previous_value=3.5,
            delta_value=0.5,
            delta_percent=14.29,
            current_date=pd.Timestamp("2025-01-01"),
            previous_date=pd.Timestamp("2024-12-01"),
            crosses_trend_break=True,
            is_missing_comparison=False,
            comparison_type="MoM",
        )
        result = format_change("u_rate", change)
        assert "trend-break" in result.break_flag

    def test_rate_kpi_no_break_flag(self):
        """No trend-break when not crossing the boundary."""
        change = KPIChange(
            current_value=3.5,
            previous_value=3.0,
            delta_value=0.5,
            delta_percent=16.67,
            current_date=pd.Timestamp("2025-02-01"),
            previous_date=pd.Timestamp("2025-01-01"),
            crosses_trend_break=False,
            is_missing_comparison=False,
            comparison_type="MoM",
        )
        result = format_change("u_rate", change)
        assert result.break_flag == ""


# ---------------------------------------------------------------------------
# format_change — count KPI
# ---------------------------------------------------------------------------

class TestFormatChangeCount:
    """Tests for format_change() with count KPIs (relative % primary)."""

    def test_count_kpi_shows_percent(self):
        change = KPIChange(
            current_value=110.0,
            previous_value=105.0,
            delta_value=5.0,
            delta_percent=4.76,
            current_date=pd.Timestamp("2025-02-01"),
            previous_date=pd.Timestamp("2025-01-01"),
            crosses_trend_break=False,
            is_missing_comparison=False,
            comparison_type="MoM",
        )
        result = format_change("lf", change)
        assert isinstance(result, ChangeDisplay)
        assert "4.8" in result.delta_str
        assert "%" in result.delta_str
        assert "pp" not in result.delta_str

    def test_count_kpi_shows_up_arrow(self):
        change = KPIChange(
            current_value=110.0,
            previous_value=105.0,
            delta_value=5.0,
            delta_percent=4.76,
            current_date=pd.Timestamp("2025-02-01"),
            previous_date=pd.Timestamp("2025-01-01"),
            crosses_trend_break=False,
            is_missing_comparison=False,
            comparison_type="MoM",
        )
        result = format_change("lf", change)
        assert result.delta_str.startswith("\u2191")
        assert result.is_favorable is True

    def test_count_kpi_shows_down_arrow(self):
        change = KPIChange(
            current_value=105.0,
            previous_value=110.0,
            delta_value=-5.0,
            delta_percent=-4.76,
            current_date=pd.Timestamp("2025-02-01"),
            previous_date=pd.Timestamp("2025-01-01"),
            crosses_trend_break=False,
            is_missing_comparison=False,
            comparison_type="MoM",
        )
        result = format_change("lf", change)
        assert result.delta_str.startswith("\u2193")
        assert result.is_favorable is True

    def test_count_kpi_no_contradictory_arrows(self):
        change = KPIChange(
            current_value=110.0,
            previous_value=105.0,
            delta_value=5.0,
            delta_percent=4.76,
            current_date=pd.Timestamp("2025-02-01"),
            previous_date=pd.Timestamp("2025-01-01"),
            crosses_trend_break=False,
            is_missing_comparison=False,
            comparison_type="MoM",
        )
        result = format_change("lf", change)
        arrow_count = result.delta_str.count("\u2191") + result.delta_str.count("\u2193")
        assert arrow_count == 1

    def test_count_kpi_no_context_str(self):
        change = KPIChange(
            current_value=110.0,
            previous_value=105.0,
            delta_value=5.0,
            delta_percent=4.76,
            current_date=pd.Timestamp("2025-02-01"),
            previous_date=pd.Timestamp("2025-01-01"),
            crosses_trend_break=False,
            is_missing_comparison=False,
            comparison_type="MoM",
        )
        result = format_change("lf", change)
        assert result.context_str == ""


# ---------------------------------------------------------------------------
# format_change — missing / edge cases
# ---------------------------------------------------------------------------

class TestFormatChangeEdgeCases:
    def test_missing_comparison_returns_na(self):
        change = KPIChange(
            current_value=3.5,
            previous_value=None,
            delta_value=None,
            delta_percent=None,
            current_date=pd.Timestamp("2025-02-01"),
            previous_date=None,
            crosses_trend_break=False,
            is_missing_comparison=True,
            comparison_type="MoM",
        )
        result = format_change("u_rate", change)
        assert isinstance(result, ChangeDisplay)
        assert result.delta_str == "N/A"
        assert result.direction == "none"

    def test_none_current_value_returns_na(self):
        change = KPIChange(
            current_value=None,
            previous_value=3.0,
            delta_value=None,
            delta_percent=None,
            current_date=pd.Timestamp("2025-02-01"),
            previous_date=pd.Timestamp("2025-01-01"),
            crosses_trend_break=False,
            is_missing_comparison=True,
            comparison_type="MoM",
        )
        result = format_change("u_rate", change)
        assert result.delta_str == "N/A"

    def test_zero_delta_shows_neutral_arrow(self):
        """Zero change shows right arrow and direction 'none'."""
        change = KPIChange(
            current_value=3.5,
            previous_value=3.5,
            delta_value=0.0,
            delta_percent=0.0,
            current_date=pd.Timestamp("2025-02-01"),
            previous_date=pd.Timestamp("2025-01-01"),
            crosses_trend_break=False,
            is_missing_comparison=False,
            comparison_type="MoM",
        )
        result = format_change("u_rate", change)
        assert "\u2192" in result.delta_str
        assert result.direction == "none"


# ---------------------------------------------------------------------------
# Favorability logic
# ---------------------------------------------------------------------------

class TestFavorability:
    def test_u_rate_decrease_is_favorable(self):
        """A decrease in unemployment rate is favorable."""
        change = KPIChange(
            current_value=2.5,
            previous_value=3.0,
            delta_value=-0.5,
            delta_percent=-16.67,
            current_date=pd.Timestamp("2025-02-01"),
            previous_date=pd.Timestamp("2025-01-01"),
            crosses_trend_break=False,
            is_missing_comparison=False,
            comparison_type="MoM",
        )
        result = format_change("u_rate", change)
        assert result.direction == "down"
        assert result.is_favorable is True

    def test_u_rate_increase_is_not_favorable(self):
        """An increase in unemployment rate is not favorable."""
        change = KPIChange(
            current_value=3.5,
            previous_value=3.0,
            delta_value=0.5,
            delta_percent=16.67,
            current_date=pd.Timestamp("2025-02-01"),
            previous_date=pd.Timestamp("2025-01-01"),
            crosses_trend_break=False,
            is_missing_comparison=False,
            comparison_type="MoM",
        )
        result = format_change("u_rate", change)
        assert result.direction == "up"
        assert result.is_favorable is False

    def test_lf_increase_is_neutral(self):
        """Labour force size change is always favorable (neutral KPI)."""
        change = KPIChange(
            current_value=110.0,
            previous_value=105.0,
            delta_value=5.0,
            delta_percent=4.76,
            current_date=pd.Timestamp("2025-02-01"),
            previous_date=pd.Timestamp("2025-01-01"),
            crosses_trend_break=False,
            is_missing_comparison=False,
            comparison_type="MoM",
        )
        result = format_change("lf", change)
        assert result.is_favorable is True

    def test_p_rate_increase_is_favorable(self):
        """Increase in participation rate is favorable."""
        change = KPIChange(
            current_value=66.0,
            previous_value=65.0,
            delta_value=1.0,
            delta_percent=1.54,
            current_date=pd.Timestamp("2025-02-01"),
            previous_date=pd.Timestamp("2025-01-01"),
            crosses_trend_break=False,
            is_missing_comparison=False,
            comparison_type="MoM",
        )
        result = format_change("p_rate", change)
        assert result.is_favorable is True

    def test_ep_ratio_decrease_is_not_favorable(self):
        """Decrease in employment-to-population ratio is not favorable."""
        change = KPIChange(
            current_value=62.0,
            previous_value=63.0,
            delta_value=-1.0,
            delta_percent=-1.59,
            current_date=pd.Timestamp("2025-02-01"),
            previous_date=pd.Timestamp("2025-01-01"),
            crosses_trend_break=False,
            is_missing_comparison=False,
            comparison_type="MoM",
        )
        result = format_change("ep_ratio", change)
        assert result.direction == "down"
        assert result.is_favorable is False


# ---------------------------------------------------------------------------
# End-to-end: compute_mom_change + format_change integration
# ---------------------------------------------------------------------------

class TestMoMFormatIntegration:
    """Integration tests: compute_mom_change results fed to format_change."""

    def test_mom_rate_arrow_direction(self):
        """Feb 2025 u_rate (3.5) vs Jan 2025 (4.0): rate decreased by 0.5pp."""
        df = _make_test_df()
        df_recent = df.tail(2)
        change = compute_mom_change(df_recent, "u_rate")

        assert change.delta_value == pytest.approx(-0.5)

        display = format_change("u_rate", change)
        assert display.delta_str.startswith("\u2193")
        assert "0.5" in display.delta_str
        assert "pp" in display.delta_str
        assert display.direction == "down"
        assert display.is_favorable is True

    def test_mom_count_arrow_direction(self):
        """Feb 2025 lf (110) vs Jan 2025 (105): increased by ~4.76%."""
        df = _make_test_df()
        df_recent = df.tail(2)
        change = compute_mom_change(df_recent, "lf")

        assert change.delta_percent == pytest.approx(4.76, rel=0.01)

        display = format_change("lf", change)
        assert display.delta_str.startswith("\u2191")
        assert "4.8" in display.delta_str
        assert display.is_favorable is True

    def test_mom_crosses_trend_break(self):
        """Jan 2025 vs Jan 2024 for u_rate crosses trend break."""
        df = _make_test_df()
        df_subset = df.iloc[[0, 2]]  # Jan 2024, Jan 2025
        change = compute_mom_change(df_subset, "u_rate")
        assert change.crosses_trend_break is True

        display = format_change("u_rate", change)
        assert "trend-break" in display.break_flag




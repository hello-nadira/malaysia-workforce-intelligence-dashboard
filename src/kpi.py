"""
KPI extraction, formatting, and comparison logic.

This module takes a validated DataFrame (from data_loader.load_dataframe)
and answers questions like:
  - "What is the latest unemployment rate?"
  - "How did it change from last month?"
  - "How does it compare to the same month last year?"

Important distinction (requirement #4):
  - For RATE KPIs (u_rate, p_rate, ep_ratio): the change is shown as a
    *percentage-point* change.  e.g. if u_rate went from 3.0 to 3.2,
    the change is "+0.2 pp" (2 percentage points), NOT "+6.7%".
  - For COUNT KPIs (lf, lf_employed, etc.): the change is shown as a
    *relative percentage* change.  e.g. if lf went from 17,000 to
    17,200, the change is "+1.2%".

This module contains NO Streamlit imports. It is pure Python + pandas
and can be tested with pytest without a web server.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import config


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class KPIValue:
    """A single KPI value at a point in time."""
    value: float | None
    date: pd.Timestamp | None
    is_missing: bool


@dataclass
class KPIChange:
    """
    A comparison between two KPI values.

    For rate KPIs, delta_value is in percentage points.
    For count KPIs, delta_percent is the relative percentage change.
    """
    current_value: float | None
    previous_value: float | None
    delta_value: float | None          # absolute difference (for rates: pp; for counts: raw)
    delta_percent: float | None        # relative % change
    current_date: pd.Timestamp | None
    previous_date: pd.Timestamp | None
    crosses_trend_break: bool          # True if comparison spans Jan 2025
    is_missing_comparison: bool        # True if either value is missing
    comparison_type: str               # "MoM" or "YoY"






# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_unit(kpi_key: str) -> str:
    """Return the unit string for a KPI key (e.g. '%' or 'K persons')."""
    return config.KPI_DEFINITIONS[kpi_key]["unit"]


def _get_column(kpi_key: str) -> str:
    """Return the DataFrame column name for a KPI key."""
    return config.KPI_DEFINITIONS[kpi_key]["column"]


def _is_rate_kpi(kpi_key: str) -> bool:
    """True for rate KPIs where changes are percentage-point based."""
    return kpi_key in config.RATE_KPI_KEYS


# ---------------------------------------------------------------------------
# 1. Latest KPI value
# ---------------------------------------------------------------------------

def get_latest_kpi(df: pd.DataFrame, kpi_key: str) -> KPIValue:
    """
    Extract the most recent value for a KPI from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        A validated DataFrame (sorted ascending by date).
    kpi_key : str
        One of the keys in config.KPI_DEFINITIONS (e.g. "u_rate").

    Returns
    -------
    KPIValue with the value, date, and missing flag.
    """
    if kpi_key not in config.KPI_DEFINITIONS:
        raise KeyError(
            f"Unknown KPI key: '{kpi_key}'. "
            f"Available: {list(config.KPI_DEFINITIONS.keys())}"
        )

    column = _get_column(kpi_key)

    if df.empty:
        return KPIValue(value=None, date=None, is_missing=True)

    latest_row = df.iloc[-1]
    value = latest_row[column]

    if pd.isna(value):
        return KPIValue(value=None, date=latest_row[config.DATE_COLUMN], is_missing=True)

    return KPIValue(
        value=float(value),
        date=latest_row[config.DATE_COLUMN],
        is_missing=False,
    )


# ---------------------------------------------------------------------------
# 2. Formatting
# ---------------------------------------------------------------------------

def format_kpi(value: float | None, unit: str) -> str:
    """
    Format a KPI value with its unit.

    - For '%' units: show with 1 decimal place (e.g. "3.1%")
    - For 'K persons' units: show with thousands separator (e.g. "17,386.0K persons")
    - For None: return "N/A"

    Parameters
    ----------
    value : float or None
        The raw numeric value.
    unit : str
        The unit string from config (e.g. "%", "K persons")

    Returns
    -------
    str : formatted value
    """
    if value is None or pd.isna(value):
        return "N/A"

    if unit == "%":
        return f"{value:.1f}%"
    elif unit == "K persons":
        return f"{value:,.1f}K persons"
    else:
        return f"{value:,.1f} {unit}"


def format_change(kpi_key: str, change: KPIChange) -> str:
    """
    Format a KPIChange for display.

    For rate KPIs:
      - Shows delta in percentage points: "A 0.2 pp" or "V 0.3 pp"
      - Also shows relative % change: "(+0.7%)"
      - Appends "*trend-break" if the comparison crosses Jan 2025

    For count KPIs:
      - Shows relative % change: "A 1.2%" or "V 0.5%"

    Parameters
    ----------
    kpi_key : str
    change : KPIChange

    Returns
    -------
    str : e.g. "A 0.2 pp (+0.7%)", "V 0.3 pp (-1.0%)", "A 1.2%", "N/A"
    """
    if (change.is_missing_comparison
            or change.current_value is None
            or change.previous_value is None):
        return "N/A"

    # Use Unicode arrows: up arrow = increase, down arrow = decrease
    if _is_rate_kpi(kpi_key):
        delta_pp = change.delta_value
        if delta_pp is None or pd.isna(delta_pp):
            return "N/A"

        if delta_pp > 0:
            direction = "\u2191"   # ↑
        elif delta_pp < 0:
            direction = "\u2193"   # ↓
        else:
            direction = "\u2192"   # →

        # Also compute relative % change for context
        rel_pct_str = ""
        if (change.delta_percent is not None
                and not pd.isna(change.delta_percent)
                and change.previous_value != 0):
            sign = "+" if change.delta_percent >= 0 else ""
            rel_pct_str = f" ({sign}{change.delta_percent:+.1f}%)"

        # Trend-break warning
        break_flag = ""
        if change.crosses_trend_break:
            break_flag = " *trend-break*"

        return f"{direction} {abs(delta_pp):.1f} pp{rel_pct_str}{break_flag}"

    else:
        if change.delta_percent is None or pd.isna(change.delta_percent):
            return "N/A"

        if change.delta_percent > 0:
            direction = "\u2191"   # ↑
        elif change.delta_percent < 0:
            direction = "\u2193"   # ↓
        else:
            direction = "\u2192"   # →

        sign = "+" if change.delta_percent >= 0 else ""
        return f"{direction} {sign}{abs(change.delta_percent):.1f}%"


# ---------------------------------------------------------------------------
# 3. Trend-break detection
# ---------------------------------------------------------------------------

def crosses_trend_break(
    current_date: pd.Timestamp | None,
    comparison_date: pd.Timestamp | None,
) -> bool:
    """
    Check whether a comparison spans the 2025 population-reference break.

    The break occurs at January 2025:
      - Before 2025: data uses 2010 Census population benchmarks
      - From 2025 onward: data uses 2020 Census population benchmarks

    Returns True if the two dates are on opposite sides of the break.

    Parameters
    ----------
    current_date : pd.Timestamp or None
        The later date (e.g. the current month).
    comparison_date : pd.Timestamp or None
        The earlier comparison date (e.g. previous month or same month last year).

    Returns
    -------
    bool
        True if current_date >= Jan 2025 AND comparison_date < Jan 2025.
    """
    if current_date is None or comparison_date is None:
        return False

    break_ts = pd.Timestamp(config.POP_REF_BREAK_DATE)

    current_is_post = current_date >= break_ts
    comparison_is_pre = comparison_date < break_ts

    return current_is_post and comparison_is_pre


# ---------------------------------------------------------------------------
# 4. Month-on-month change
# ---------------------------------------------------------------------------

def compute_mom_change(df: pd.DataFrame, kpi_key: str) -> KPIChange:
    """
    Compute the month-on-month change for a KPI.

    Compares the latest value to the value one row earlier (previous month).
    For rate KPIs, delta_value is in percentage points.
    For count KPIs, delta_value is the raw difference (in thousands).

    Returns a KPIChange dataclass. If there is only one data point,
    or either value is missing, is_missing_comparison is True.
    """
    column = _get_column(kpi_key)

    if len(df) < 2:
        current_val = df.iloc[-1][column] if len(df) == 1 else None
        return KPIChange(
            current_value=float(current_val) if pd.notna(current_val) else None,
            previous_value=None,
            delta_value=None,
            delta_percent=None,
            current_date=df.iloc[-1][config.DATE_COLUMN] if len(df) == 1 else None,
            previous_date=None,
            crosses_trend_break=False,
            is_missing_comparison=True,
            comparison_type="MoM",
        )

    latest_row = df.iloc[-1]
    prev_row = df.iloc[-2]

    current_value = latest_row[column]
    previous_value = prev_row[column]
    current_date = latest_row[config.DATE_COLUMN]
    previous_date = prev_row[config.DATE_COLUMN]

    # Check for missing values
    if pd.isna(current_value) or pd.isna(previous_value):
        return KPIChange(
            current_value=float(current_value) if pd.notna(current_value) else None,
            previous_value=float(previous_value) if pd.notna(previous_value) else None,
            delta_value=None,
            delta_percent=None,
            current_date=current_date,
            previous_date=previous_date,
            crosses_trend_break=crosses_trend_break(current_date, previous_date),
            is_missing_comparison=True,
            comparison_type="MoM",
        )

    current_value = float(current_value)
    previous_value = float(previous_value)

    if _is_rate_kpi(kpi_key):
        # Percentage-point difference
        delta = current_value - previous_value
        # Relative change for context
        delta_pct = (
            ((current_value - previous_value) / previous_value * 100)
            if previous_value != 0 else None
        )
    else:
        # Count KPI: raw difference (thousands), and relative %
        delta = current_value - previous_value
        delta_pct = (
            ((current_value - previous_value) / previous_value * 100)
            if previous_value != 0 else None
        )

    return KPIChange(
        current_value=current_value,
        previous_value=previous_value,
        delta_value=delta,
        delta_percent=delta_pct,
        current_date=current_date,
        previous_date=previous_date,
        crosses_trend_break=crosses_trend_break(current_date, previous_date),
        is_missing_comparison=False,
        comparison_type="MoM",
    )


# ---------------------------------------------------------------------------
# 5. Year-on-year change
# ---------------------------------------------------------------------------

def compute_yoy_change(df: pd.DataFrame, kpi_key: str) -> KPIChange:
    """
    Compute the year-on-year change for a KPI.

    Compares the latest value to the value from the same month one year
    earlier.  If the exact same month is not available, finds the closest
    available month in the prior year's data.

    Returns a KPIChange dataclass. If no comparison data exists,
    is_missing_comparison is True.
    """
    column = _get_column(kpi_key)

    if df.empty:
        return KPIChange(
            current_value=None,
            previous_value=None,
            delta_value=None,
            delta_percent=None,
            current_date=None,
            previous_date=None,
            crosses_trend_break=False,
            is_missing_comparison=True,
            comparison_type="YoY",
        )

    latest_row = df.iloc[-1]
    current_value = latest_row[column]
    current_date = latest_row[config.DATE_COLUMN]

    # Calculate the target comparison date (same month, one year prior)
    target_date = current_date - pd.DateOffset(years=1)

    # Find the row closest to the target date
    # First try exact match, then fall back to nearest
    exact_match = df[df[config.DATE_COLUMN] == target_date]

    if len(exact_match) == 1:
        comparison_row = exact_match.iloc[0]
    else:
        # Find nearest available date in the prior year
        prior_year_data = df[df[config.DATE_COLUMN] < current_date]
        same_month_prior = prior_year_data[
            prior_year_data[config.DATE_COLUMN].dt.month == target_date.month
        ]
        if len(same_month_prior) > 0:
            # Same month in prior year (may be from a different actual year)
            comparison_row = same_month_prior.iloc[-1]
        else:
            # No same-month data available
            return KPIChange(
                current_value=float(current_value) if pd.notna(current_value) else None,
                previous_value=None,
                delta_value=None,
                delta_percent=None,
                current_date=current_date,
                previous_date=None,
                crosses_trend_break=crosses_trend_break(current_date, None),
                is_missing_comparison=True,
                comparison_type="YoY",
            )

    previous_value = comparison_row[column]
    previous_date = comparison_row[config.DATE_COLUMN]

    if pd.isna(current_value) or pd.isna(previous_value):
        return KPIChange(
            current_value=float(current_value) if pd.notna(current_value) else None,
            previous_value=float(previous_value) if pd.notna(previous_value) else None,
            delta_value=None,
            delta_percent=None,
            current_date=current_date,
            previous_date=previous_date,
            crosses_trend_break=crosses_trend_break(current_date, previous_date),
            is_missing_comparison=True,
            comparison_type="YoY",
        )

    current_value = float(current_value)
    previous_value = float(previous_value)

    if _is_rate_kpi(kpi_key):
        delta = current_value - previous_value
        delta_pct = (
            ((current_value - previous_value) / previous_value * 100)
            if previous_value != 0 else None
        )
    else:
        delta = current_value - previous_value
        delta_pct = (
            ((current_value - previous_value) / previous_value * 100)
            if previous_value != 0 else None
        )

    return KPIChange(
        current_value=current_value,
        previous_value=previous_value,
        delta_value=delta,
        delta_percent=delta_pct,
        current_date=current_date,
        previous_date=previous_date,
        crosses_trend_break=crosses_trend_break(current_date, previous_date),
        is_missing_comparison=False,
        comparison_type="YoY",
    )


# ---------------------------------------------------------------------------
# 6. Date-range filtering
# ---------------------------------------------------------------------------

def filter_by_date_range(
    df: pd.DataFrame,
    start_date: pd.Timestamp | str | None,
    end_date: pd.Timestamp | str | None,
) -> pd.DataFrame:
    """
    Filter the DataFrame to a date range (inclusive on both ends).

    Parameters
    ----------
    df : pd.DataFrame
        Validated DataFrame with a parsed date column.
    start_date : pd.Timestamp, str, or None
        Start of the range (inclusive). If None, no lower bound.
    end_date : pd.Timestamp, str, or None
        End of the range (inclusive). If None, no upper bound.

    Returns
    -------
    pd.DataFrame filtered to the range, sorted ascending by date.
    """
    mask = pd.Series([True] * len(df), index=df.index)

    if start_date is not None:
        start_ts = pd.Timestamp(start_date)
        mask &= df[config.DATE_COLUMN] >= start_ts

    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        mask &= df[config.DATE_COLUMN] <= end_ts

    return df[mask].sort_values(config.DATE_COLUMN, ascending=True).reset_index(drop=True)


def get_available_date_range(
    df: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (earliest_date, latest_date) available in the DataFrame."""
    return (df[config.DATE_COLUMN].min(), df[config.DATE_COLUMN].max())





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


@dataclass
class ChangeDisplay:
    """
    Formatted output of format_change(), ready for display.

    Attributes
    ----------
    delta_str : str
        Primary directional string, e.g. "↓ 0.1 pp", "↑ +1.2%", "N/A"
    delta_str_no_arrow : str
        Same as delta_str but with the leading arrow removed.
        Use this when passing to st.metric's delta parameter to avoid
        double arrows (our arrow + Streamlit's automatic arrow).
    context_str : str
        Secondary labeled context for rate KPIs, e.g. "rel: +3.3%", or ""
    direction : str
        Internal direction code: "up", "down", or "none"
    is_favorable : bool
        Whether the direction is generally favorable for this KPI
    break_flag : str
        " *trend-break*" if comparison crosses the Jan 2025 break, else ""
    """
    delta_str: str
    delta_str_no_arrow: str
    context_str: str
    direction: str
    is_favorable: bool
    break_flag: str


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
_ARROW_MAP = {"up": "\u2191", "down": "\u2193", "none": "\u2192"}


def _arrow(direction: str) -> str:
    """Return the Unicode arrow for a direction string."""
    return _ARROW_MAP.get(direction, "\u2192")


def get_favorability(kpi_key: str) -> str:
    """
    Return the favourability direction for a KPI.

    Returns "up" if an increase is favourable, "down" if a decrease
    is favourable, or "neutral" if neither direction is inherently
    good or bad.
    """
    return config.KPI_FAVORABILITY.get(kpi_key, "neutral")


def is_favorable_change(kpi_key: str, direction: str) -> bool:
    """
    Determine whether a directional change is favourable for a KPI.

    Parameters
    ----------
    kpi_key : str
        The KPI key (e.g. "u_rate", "lf").
    direction : str
        One of "up", "down", or "none".

    Returns
    -------
    bool
        True if the direction is favourable for this KPI.
        Neutral KPIs always return True (no color coding).
    """
    favorability = get_favorability(kpi_key)

    if favorability == "neutral":
        return True  # Don't color-code neutral KPIs

    if direction == "none":
        return True

    return direction == favorability


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
    - For 'K persons' units: show in readable form (e.g. "17.4M people")
    - For None: return "N/A"

    Parameters
    ----------
    value : float or None
        The raw numeric value (in the unit specified).
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
        # Convert thousands to a more readable format
        # e.g. 17386.0 (thousands) -> "17.4M people"
        millions = value / 1000.0
        return f"{millions:.1f}M people"
    else:
        # Generic fallback for unknown units
        return f"{value:,.1f} {unit}"


def get_favorability(kpi_key: str) -> str:
    """
    Return the favourability direction for a KPI.

    Returns "up" if an increase is favourable, "down" if a decrease
    is favourable, or "neutral" if neither direction is inherently
    good or bad.
    """
    return config.KPI_FAVORABILITY.get(kpi_key, "neutral")


def is_favorable_change(kpi_key: str, direction: str) -> bool:
    """
    Determine whether a directional change is favourable for a KPI.

    Parameters
    ----------
    kpi_key : str
        The KPI key (e.g. "u_rate", "lf").
    direction : str
        One of "up", "down", or "none".

    Returns
    -------
    bool
        True if the direction is favourable for this KPI.
        Neutral KPIs always return True (no color coding).
    """
    favorability = get_favorability(kpi_key)

    if favorability == "neutral":
        return True  # Don't color-code neutral KPIs

    if direction == "none":
        return True

    return direction == favorability


def format_change(kpi_key: str, change: KPIChange) -> ChangeDisplay:
    """
    Format a KPIChange for display.

    Returns a ChangeDisplay dataclass with:
    - delta_str: The primary displayed string (e.g. "↓ 0.1 pp", "↑ +1.2%")
    - context_str: Secondary labeled context (e.g. "rel: +3.3%") or ""
    - direction: "up", "down", or "none"
    - is_favorable: Whether the change is generally favorable for this KPI
    - break_flag: " *trend-break*" if comparison crosses Jan 2025, else ""

    For rate KPIs:
    - delta_str shows percentage-point change (primary)
    - context_str shows relative % change labeled as "rel:"

    For count KPIs:
    - delta_str shows relative % change
    - context_str is empty (no secondary metric)

    If comparison is missing, returns a ChangeDisplay with delta_str="N/A".
    """
    na_display = ChangeDisplay(
        delta_str="N/A",
        delta_str_no_arrow="N/A",
        context_str="",
        direction="none",
        is_favorable=True,
        break_flag="",
    )

    if (change.is_missing_comparison
            or change.current_value is None
            or change.previous_value is None):
        return na_display

    # Trend-break warning
    break_flag = " *trend-break*" if change.crosses_trend_break else ""

    if _is_rate_kpi(kpi_key):
        delta_pp = change.delta_value
        if delta_pp is None or pd.isna(delta_pp):
            return na_display

        # Single arrow based on the primary metric (pp change)
        if delta_pp > 0:
            direction = "up"
        elif delta_pp < 0:
            direction = "down"
        else:
            direction = "none"

        favorable = is_favorable_change(kpi_key, direction)

        arrow = _arrow(direction)
        delta_str = f"{arrow} {abs(delta_pp):.1f} pp"
        delta_no_arrow = f"{abs(delta_pp):.1f} pp"

        # Secondary: relative % change labeled as "rel:"
        context_str = ""
        if (change.delta_percent is not None
                and not pd.isna(change.delta_percent)
                and change.previous_value != 0):
            context_str = f"rel: {change.delta_percent:+.1f}%"

        return ChangeDisplay(
            delta_str=delta_str,
            delta_str_no_arrow=delta_no_arrow,
            context_str=context_str,
            direction=direction,
            is_favorable=favorable,
            break_flag=break_flag,
        )

    else:
        # Count KPI
        if change.delta_percent is None or pd.isna(change.delta_percent):
            return na_display

        if change.delta_percent > 0:
            direction = "up"
        elif change.delta_percent < 0:
            direction = "down"
        else:
            direction = "none"

        favorable = is_favorable_change(kpi_key, direction)

        arrow = _arrow(direction)
        delta_str = f"{arrow} {change.delta_percent:+.1f}%"

        # For st.metric: strip the arrow, let Streamlit render its own
        # The sign in delta_percent already indicates direction
        if change.delta_percent > 0:
            delta_no_arrow = f"+{change.delta_percent:.1f}%"
        elif change.delta_percent < 0:
            delta_no_arrow = f"{change.delta_percent:.1f}%"
        else:
            delta_no_arrow = f"{change.delta_percent:.1f}%"

        return ChangeDisplay(
            delta_str=delta_str,
            delta_str_no_arrow=delta_no_arrow,
            context_str="",
            direction=direction,
            is_favorable=favorable,
            break_flag=break_flag,
        )


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





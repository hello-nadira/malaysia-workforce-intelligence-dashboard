"""
Executive insights generation (rule-based, deterministic).

This module examines the latest KPI values and their MoM/YoY changes
and produces a short list of evidence-based observations.

IMPORTANT DESIGN PRINCIPLES (requirement #8):
  - No causal claims.  We describe *what* changed, not *why*.
  - No "healthy"/"unhealthy" labels without a defined methodology.
  - We explicitly flag when a comparison crosses the 2025 population-
    reference break so the reader knows the change may be driven by
    a revised population denominator.

This module contains NO Streamlit imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import config
from src.kpi import (
    compute_mom_change,
    compute_yoy_change,
    format_change,
    get_latest_kpi,
)


@dataclass
class Insight:
    """A single executive insight bullet point."""
    text: str
    is_trend_break_warning: bool = False
    priority: int = 0  # higher = more important (sorted first)


def generate_insights(
    df: pd.DataFrame,
) -> list[Insight]:
    """
    Generate a list of rule-based executive insights from the latest data.

    Parameters
    ----------
    df : pd.DataFrame
        A validated, parsed DataFrame (from data_loader.load_dataframe).

    Returns
    -------
    list[Insight], sorted by priority (highest first).

    Beginner note: Each rule is a simple "if this pattern is detected in
    the data, generate this text" statement.  No machine learning or LLMs.
    """
    insights: list[Insight] = []

    if df.empty:
        insights.append(Insight(
            text="No data available to generate insights.",
            priority=10,
        ))
        return insights

    # --- Insight: Latest data point ---
    latest_date = df[config.DATE_COLUMN].max()
    insights.append(Insight(
        text=f"The latest data point is for {latest_date.strftime('%B %Y')}.",
        priority=10,
    ))

    # --- Insights for each display KPI ---
    for kpi_key in config.DISPLAY_KPI_KEYS:
        kpi_def = config.KPI_DEFINITIONS[kpi_key]
        latest = get_latest_kpi(df, kpi_key)
        mom = compute_mom_change(df, kpi_key)
        yoy = compute_yoy_change(df, kpi_key)

        if latest.is_missing:
            insights.append(Insight(
                text=f"No value available for {kpi_def['label']} in the latest month.",
                priority=5,
            ))
            continue

        # --- MoM observation ---
        if not mom.is_missing_comparison and mom.delta_value is not None:
            direction = "increased" if mom.delta_value > 0 else (
                "decreased" if mom.delta_value < 0 else "remained unchanged"
            )

            if kpi_key in config.RATE_KPI_KEYS:
                insight_text = (
                    f"{kpi_def['label']} was {latest.value:.1f}{kpi_def['unit']} "
                    f"in {latest_date.strftime('%B %Y')}, {direction} by "
                    f"{abs(mom.delta_value):.1f} percentage points "
                    f"compared to the previous month "
                    f"({_format_comparison_date(mom.previous_date)})."
                )
            else:
                insight_text = (
                    f"{kpi_def['label']} was {latest.value:,.1f}{kpi_def['unit']} "
                    f"in {latest_date.strftime('%B %Y')}, {direction} by "
                    f"{abs(mom.delta_percent):.1f}% compared to the previous month "
                    f"({_format_comparison_date(mom.previous_date)})."
                )

            is_warning = mom.crosses_trend_break
            if is_warning:
                insight_text += (
                    " Note: this comparison crosses the January 2025 "
                    "population-reference break (2010 to 2020 Census). "
                    "The observed change may reflect the revised population "
                    "denominator, not necessarily an economic shift."
                )

            insights.append(Insight(
                text=insight_text,
                is_trend_break_warning=is_warning,
                priority=3 if is_warning else 1,
            ))

        # --- YoY observation ---
        if not yoy.is_missing_comparison and yoy.delta_value is not None:
            direction = "increased" if yoy.delta_value > 0 else (
                "decreased" if yoy.delta_value < 0 else "remained unchanged"
            )

            if kpi_key in config.RATE_KPI_KEYS:
                insight_text = (
                    f"Compared to {yoy.previous_date.strftime('%B %Y')} "
                    f"({_years_ago_label(yoy.previous_date, latest_date)}), "
                    f"{kpi_def['label']} {direction} by "
                    f"{abs(yoy.delta_value):.1f} percentage points."
                )
            else:
                insight_text = (
                    f"Compared to {yoy.previous_date.strftime('%B %Y')} "
                    f"({_years_ago_label(yoy.previous_date, latest_date)}), "
                    f"{kpi_def['label']} {direction} by "
                    f"{abs(yoy.delta_percent):.1f}%."
                )

            is_warning = yoy.crosses_trend_break
            if is_warning:
                insight_text += (
                    " Note: this year-on-year comparison crosses the January "
                    "2025 population-reference break. Interpret with caution."
                )

            insights.append(Insight(
                text=insight_text,
                is_trend_break_warning=is_warning,
                priority=2 if is_warning else 0,
            ))

    # --- Insight: Data range available ---
    earliest_date = df[config.DATE_COLUMN].min()
    insights.append(Insight(
        text=(
            f"Historical data spans {earliest_date.strftime('%B %Y')} "
            f"to {latest_date.strftime('%B %Y')} ({len(df)} monthly observations)."
        ),
        priority=1,
    ))

    # Sort by priority descending (most important first)
    insights.sort(key=lambda i: -i.priority)

    return insights


def _format_comparison_date(dt: pd.Timestamp | None) -> str:
    """Format a date for use in a comparison description."""
    if dt is None:
        return "an earlier month"
    return dt.strftime("%B %Y")


def _years_ago_label(prev_date: pd.Timestamp, current_date: pd.Timestamp) -> str:
    """Return a human-readable label for how many years ago."""
    delta_years = current_date.year - prev_date.year
    if delta_years == 1:
        return "one year ago"
    return f"{delta_years} years ago"

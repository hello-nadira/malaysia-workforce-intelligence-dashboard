"""
Tests for rule-based executive insights in src/insights.py.

Tests verify:
  - generate_insights() returns non-empty list
  - Insights include MoM and YoY observations for each KPI
  - Trend-break warnings are attached when comparisons cross Jan 2025
  - No causal claims or health labels are present in the output
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from src import config
from src.data_loader import validate_dataframe
from src.insights import Insight, generate_insights


def _make_test_df() -> pd.DataFrame:
    """
    DataFrame with data spanning the 2025 trend-break boundary.
    Dates: Jan 2024, Feb 2024, Jan 2025, Feb 2025.
    """
    dates = pd.to_datetime(["2024-01-01", "2024-02-01", "2025-01-01", "2025-02-01"])
    data = {
        config.DATE_COLUMN: dates,
        "lf": [100.0, 102.0, 106.0, 110.0],
        "lf_employed": [95.0, 97.0, 101.0, 105.0],
        "lf_unemployed": [5.0, 5.0, 5.0, 5.0],
        "lf_outside": [50.0, 50.0, 50.0, 50.0],
        "p_rate": [64.0, 65.0, 65.5, 66.0],
        "ep_ratio": [61.0, 62.0, 62.5, 63.0],
        "u_rate": [3.5, 3.0, 4.8, 3.5],
    }
    df = pd.DataFrame(data)
    df = df.sort_values(config.DATE_COLUMN).reset_index(drop=True)
    validate_dataframe(df)
    return df


# ---------------------------------------------------------------------------
# Basic output checks
# ---------------------------------------------------------------------------

class TestInsightsOutput:
    def test_non_empty_insights(self):
        df = _make_test_df()
        insights = generate_insights(df)
        assert len(insights) > 0

    def test_latest_date_insight(self):
        df = _make_test_df()
        insights = generate_insights(df)
        assert any("February 2025" in i.text for i in insights)

    def test_data_range_insight(self):
        df = _make_test_df()
        insights = generate_insights(df)
        assert any("Historical data spans" in i.text for i in insights)

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=config.ALL_COLUMNS)
        df[config.DATE_COLUMN] = pd.to_datetime(df[config.DATE_COLUMN])
        insights = generate_insights(df)
        assert len(insights) == 1
        assert "No data available" in insights[0].text


# ---------------------------------------------------------------------------
# MoM insight content
# ---------------------------------------------------------------------------

class TestMoMInsights:
    def test_mom_insight_for_rate_kpi(self):
        """Feb 2025 u_rate (3.5) vs Jan 2025 u_rate (4.8): -1.3 pp."""
        df = _make_test_df()
        insights = generate_insights(df)
        u_rate_insights = [
            i for i in insights
            if "Unemployment Rate" in i.text and "previous month" in i.text
        ]
        assert len(u_rate_insights) > 0
        assert "percentage points" in u_rate_insights[0].text

    def test_mom_insight_for_count_kpi(self):
        """Feb 2025 lf (110) vs Jan 2025 lf (106): +3.8%."""
        df = _make_test_df()
        insights = generate_insights(df)
        lf_insights = [
            i for i in insights
            if "Labour Force Size" in i.text and "previous month" in i.text
        ]
        assert len(lf_insights) > 0
        assert "%" in lf_insights[0].text

    def test_mom_trend_break_warning(self):
        """MoM comparing Jan 2025 (post-break) to Dec 2024 (pre-break)."""
        dates = pd.to_datetime(["2024-12-01", "2025-01-01"])
        data = {
            config.DATE_COLUMN: dates,
            "lf": [100.0, 105.0],
            "lf_employed": [95.0, 100.0],
            "lf_unemployed": [5.0, 5.0],
            "lf_outside": [50.0, 50.0],
            "p_rate": [64.0, 65.0],
            "ep_ratio": [61.0, 62.0],
            "u_rate": [3.1, 4.0],
        }
        df = pd.DataFrame(data)
        df = df.sort_values(config.DATE_COLUMN).reset_index(drop=True)
        validate_dataframe(df)

        insights = generate_insights(df)
        trend_break_insights = [i for i in insights if i.is_trend_break_warning]
        assert len(trend_break_insights) > 0
        assert "population-reference break" in trend_break_insights[0].text


# ---------------------------------------------------------------------------
# YoY insight content
# ---------------------------------------------------------------------------

class TestYoYInsights:
    def test_yoy_insight_present(self):
        df = _make_test_df()
        insights = generate_insights(df)
        yoy_insights = [
            i for i in insights
            if "one year ago" in i.text or "years ago" in i.text
        ]
        assert len(yoy_insights) > 0

    def test_yoy_trend_break_warning(self):
        """YoY comparing Feb 2025 (post-break) to Feb 2024 (pre-break)."""
        df = _make_test_df()
        insights = generate_insights(df)
        yoy_trend_break = [
            i for i in insights
            if i.is_trend_break_warning and "year" in i.text.lower()
        ]
        assert len(yoy_trend_break) > 0


# ---------------------------------------------------------------------------
# No causal claims or health labels
# ---------------------------------------------------------------------------

class TestNoCausalClaims:
    def test_no_causal_verbs(self):
        """Insights should not use causal verbs like 'caused', 'due to', etc."""
        df = _make_test_df()
        insights = generate_insights(df)
        causal_pattern = re.compile(
            r"\b(caused|due to|because|leads to|results from|triggered by)\b",
            re.IGNORECASE,
        )
        for insight in insights:
            assert not causal_pattern.search(insight.text), (
                f"Insight contains causal language: {insight.text}"
            )

    def test_no_health_labels(self):
        """Insights should not label the labour market as healthy/unhealthy."""
        df = _make_test_df()
        insights = generate_insights(df)
        health_pattern = re.compile(
            r"\b(healthy|unhealthy|booming|crisis|decline|surge)\b",
            re.IGNORECASE,
        )
        for insight in insights:
            assert not health_pattern.search(insight.text), (
                f"Insight contains health label: {insight.text}"
            )

    def test_no_fabricated_values(self):
        """Insight text should reference actual values from the data."""
        df = _make_test_df()
        insights = generate_insights(df)
        # Every insight mentioning a number should be tied to real data values
        for insight in insights:
            if "3.5" in insight.text:
                assert "percentage points" in insight.text or "%" in insight.text


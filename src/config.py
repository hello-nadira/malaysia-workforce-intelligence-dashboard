"""
Central configuration for the Malaysia Workforce Intelligence Dashboard.

This module holds all constants — URLs, file paths, column names, KPI
definitions, units, tooltips, and the documented population-reference
break date.  Keeping everything here means there are no "magic strings"
scattered through the rest of the code, and every module imports from a
single source of truth.

Beginner note: think of this file as the dashboard's "settings panel".
Nothing data or display logic — just the raw facts the other modules
need to agree on.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (used so paths work regardless of the working directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data source — verified against https://data.gov.my/data-catalogue/lfs_month
# ---------------------------------------------------------------------------
CSV_URL: str = "https://storage.dosm.gov.my/labour/lfs_month.csv"
SOURCE_NAME: str = "Department of Statistics Malaysia (DOSM)"
LICENSE: str = "Creative Commons Attribution 4.0 International (CC BY 4.0)"

# ---------------------------------------------------------------------------
# Disk-cache settings
# ---------------------------------------------------------------------------
CACHE_DIR: Path = PROJECT_ROOT / ".dashboard_cache"
CACHE_FILE: Path = CACHE_DIR / "lfs_month.parquet"
CACHE_TTL_SECONDS: int = 3600  # re-fetch if cache is older than 1 hour

# ---------------------------------------------------------------------------
# Expected CSV schema (verified from the live file)
# ---------------------------------------------------------------------------
DATE_COLUMN: str = "date"

NUMERIC_COLUMNS: list[str] = [
    "lf",           # Labour Force Size (thousands)
    "lf_employed",  # Employed persons (thousands)
    "lf_unemployed", # Unemployed persons (thousands)
    "lf_outside",   # Persons outside labour force (thousands)
    "p_rate",       # Participation Rate (percent)
    "ep_ratio",     # Employment-to-Population Ratio (percent)
    "u_rate",       # Unemployment Rate (percent)
]

ALL_COLUMNS: list[str] = [DATE_COLUMN] + NUMERIC_COLUMNS

# ---------------------------------------------------------------------------
# KPI definitions — each entry maps an internal key to everything the
# dashboard needs to know about that KPI: the column in the CSV, a
# human-friendly label, the unit, and an explanatory tooltip.
# ---------------------------------------------------------------------------
KPI_DEFINITIONS: dict[str, dict[str, str]] = {
    "u_rate": {
        "column": "u_rate",
        "label": "Unemployment Rate",
        "unit": "%",
        "tooltip": (
            "Percentage of the labour force that is unemployed and "
            "actively seeking work."
        ),
    },
    "lf": {
        "column": "lf",
        "label": "Labour Force Size",
        "unit": "K persons",
        "tooltip": (
            "Total number of employed and unemployed persons, in thousands."
        ),
    },
    "p_rate": {
        "column": "p_rate",
        "label": "Participation Rate",
        "unit": "%",
        "tooltip": (
            "Percentage of the working-age population (15–64) that is "
            "participating in the labour force."
        ),
    },
    "ep_ratio": {
        "column": "ep_ratio",
        "label": "Employment-to-Population Ratio",
        "unit": "%",
        "tooltip": (
            "Percentage of the working-age population (15–64) that is employed."
        ),
    },
}

# The four KPIs shown on executive cards, in display order.
DISPLAY_KPI_KEYS: list[str] = ["u_rate", "lf", "p_rate", "ep_ratio"]

# ---------------------------------------------------------------------------
# Population-reference break — documented on the DOSM site:
#   "Data prior to 2025 is compiled in concordance with population data
#    from the 2010 Census. Data from 2025 onwards is compiled in
#    concordance with population data from the 2020 Census."
#
# This causes a methodological break in the series.  Comparisons that
# span this boundary (MoM, YoY) must be flagged because the jump is
# driven by a denominator revision, not necessarily by economic change.
# ---------------------------------------------------------------------------
POP_REF_BREAK_DATE: str = "2025-01-01"  # first month using 2020 Census

# The last month using the 2010 Census (for boundary messaging).
# We store this as a string so it can be looked up in the data; the
# exact value is computed in data_loader once we have the DataFrame.
LAST_2010_CENSUS_MONTH: str = "2024-12-01"

# ---------------------------------------------------------------------------
# Rate KPIs (as opposed to count KPIs) — for these the "change" shown
# to users is a *percentage-point* change, not a relative % change.
# This distinction matters for how we label deltas in the UI.
# ---------------------------------------------------------------------------
RATE_KPI_KEYS: set[str] = {"u_rate", "p_rate", "ep_ratio"}
COUNT_KPI_KEYS: set[str] = {"lf", "lf_employed", "lf_unemployed", "lf_outside"}

# ---------------------------------------------------------------------------
# Chart / UI defaults
# ---------------------------------------------------------------------------
DEFAULT_DATE_RANGE_START: str = "2020-01-01"
DEFAULT_DATE_RANGE_END: str = "2026-08-01"  # matches latest known data
CHART_HEIGHT: int = 400

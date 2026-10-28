"""
Malaysia Workforce Intelligence Dashboard
==========================================

A Streamlit dashboard for Malaysia's Monthly Principal Labour Force
Statistics from the Department of Statistics Malaysia (DOSM).

Data source: https://data.gov.my/data-catalogue/lfs_month

Beginner note: This file is the "thin UI layer". All data loading,
validation, KPI calculation, and insight generation live in src/
modules. This file imports them and renders the Streamlit interface.

Run:
    streamlit run dashboard.py
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so 'src' is importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.data_loader import (
    DataDownloadError,
    DataValidationError,
    clear_cache,
    is_cache_expired,
    load_dataframe,
)
from src.insights import generate_insights
from src.kpi import (
    compute_mom_change,
    compute_yoy_change,
    filter_by_date_range,
    format_change,
    format_kpi,
    get_available_date_range,
    get_latest_kpi,
    get_favorability,
)
from src.metadata import (
    get_data_summary_text,
    get_latest_observation_date,
    get_source_attribution,
    get_source_url,
)
from src.data_loader import CacheMetadata


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Malaysia Workforce Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data loading (with error handling)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data...")
def _load_cached_data(force_refresh: bool = False) -> tuple[pd.DataFrame, CacheMetadata, list[str]]:
    """Wrapper for load_dataframe that Streamlit can cache."""
    return load_dataframe(force_refresh=force_refresh)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    refresh = st.button("🔄 Refresh Data", type="primary")
    if refresh:
        st.cache_data.clear()
        st.rerun()

    st.divider()

    # Date range
    st.subheader("📅 Date Range")
    # Get available range from cached data
    sidebar_avail_start = None
    sidebar_avail_end = None
    sidebar_default_start = pd.Timestamp(config.DEFAULT_DATE_RANGE_START)
    sidebar_default_end = pd.Timestamp(config.DEFAULT_DATE_RANGE_END)
    try:
        df_preview, _, _ = _load_cached_data(force_refresh=False)
        sidebar_avail_start, sidebar_avail_end = get_available_date_range(df_preview)
        sidebar_default_start = max(
            pd.Timestamp(config.DEFAULT_DATE_RANGE_START),
            sidebar_avail_start,
        )
        sidebar_default_end = sidebar_avail_end
    except Exception:
        pass

    date_range = st.date_input(
        "Select date range",
        value=(sidebar_default_start.date(), sidebar_default_end.date()),
        min_value=sidebar_avail_start.date() if sidebar_avail_start else None,
        max_value=sidebar_avail_end.date() if sidebar_avail_end else None,
    )

    st.divider()

    # Metric selector
    st.subheader("📊 Metric")
    default_metric = "u_rate"
    metric_key = st.selectbox(
        "Select metric for trend chart",
        options=config.DISPLAY_KPI_KEYS,
        format_func=lambda k: config.KPI_DEFINITIONS[k]["label"],
        index=config.DISPLAY_KPI_KEYS.index(default_metric),
    )

    st.divider()
    st.caption(f"License: {config.LICENSE}")


# ---------------------------------------------------------------------------
# Main: Load data
# ---------------------------------------------------------------------------
try:
    df, metadata, load_messages = _load_cached_data(force_refresh=False)
except DataDownloadError:
    st.error(
        "Unable to download data from the official source and no cached "
        "data is available. Please check your internet connection and "
        "try clicking 'Refresh Data' again."
    )
    st.stop()
except DataValidationError as e:
    st.error(f"Data validation failed: {e}")
    st.stop()


# ---------------------------------------------------------------------------
# Status messages
# ---------------------------------------------------------------------------
for msg in load_messages:
    if "WARNING" in msg or "cached" in msg.lower():
        st.warning(msg, icon="⚠️")
    else:
        st.info(msg, icon="ℹ️")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📊 Malaysia Workforce Intelligence Dashboard")
st.markdown(
    f"**Source:** {get_source_attribution()}  |  "
    f"[Direct CSV]({get_source_url()})"
)


# ---------------------------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------------------------
latest_obs = get_latest_observation_date(df)
available_start, available_end = get_available_date_range(df)

st.subheader(f"📈 Key Metrics (Latest: {latest_obs.strftime('%B %Y')})")

# Use columns for 4 KPI cards
cols = st.columns(4)

for i, kpi_key in enumerate(config.DISPLAY_KPI_KEYS):
    kpi_def = config.KPI_DEFINITIONS[kpi_key]
    kpi_value = get_latest_kpi(df, kpi_key)
    mom_change = compute_mom_change(df, kpi_key)
    yoy_change = compute_yoy_change(df, kpi_key)

    unit = kpi_def["unit"]
    formatted_value = format_kpi(kpi_value.value, unit)
    mom_display = format_change(kpi_key, mom_change)
    yoy_display = format_change(kpi_key, yoy_change)

    # Show observation month alongside the metric label
    obs_label = f"{kpi_def['label']} — {latest_obs.strftime('%b %Y')}"

    with cols[i]:
        # st.metric draws its OWN arrow when we pass a numeric string as delta.
        # We pass delta_str_no_arrow (without our manual arrow) to avoid
        # double arrows. Then we control direction via delta_arrow and
        # color via delta_color based on KPI favourability:
        #   - "normal": green for positive, red for negative (default)
        #   - "inverse": green for negative, red for positive (decrease is good)
        #   - "off": gray regardless (neutral KPIs like labour force size)
        favorability = get_favorability(kpi_key)
        if favorability == "neutral":
            delta_color = "off"
        elif favorability == "down":
            delta_color = "inverse"  # green when decrease (negative delta)
        else:
            delta_color = "normal"   # green when increase (positive delta)

        # Map internal direction to Streamlit's delta_arrow values
        if mom_display.direction == "up":
            delta_arrow = "up"
        elif mom_display.direction == "down":
            delta_arrow = "down"
        else:
            delta_arrow = "right"

        if mom_display.delta_str == "N/A":
            st.metric(
                label=obs_label,
                value=formatted_value,
                delta=None,
                delta_color="off",
            )
        else:
            st.metric(
                label=obs_label,
                value=formatted_value,
                delta=mom_display.delta_str_no_arrow,
                delta_color=delta_color,
                delta_arrow=delta_arrow,
            )

        # Build the secondary line: context + trend-break flag
        secondary_parts = []
        if mom_display.context_str:
            secondary_parts.append(mom_display.context_str)
        if mom_display.break_flag.strip():
            secondary_parts.append(mom_display.break_flag.strip())
        secondary_text = " ".join(secondary_parts) if secondary_parts else ""

        if secondary_text:
            st.caption(f"ℹ️ {secondary_text}")

        # MoM and YoY with consistent labels — arrow comes from delta_str
        st.caption(f"MoM: {mom_display.delta_str}" + mom_display.break_flag)
        st.caption(f"YoY: {yoy_display.delta_str}" + yoy_display.break_flag)


# ---------------------------------------------------------------------------
# Trend chart with date-range filter
# ---------------------------------------------------------------------------
st.subheader("📉 Trend Chart")

# Apply date range filter
start_date = pd.Timestamp(date_range[0]) if date_range else available_start
end_date = pd.Timestamp(date_range[1]) if date_range else available_end
filtered_df = filter_by_date_range(df, start_date, end_date)

# Get the KPI column and label
chart_column = config.KPI_DEFINITIONS[metric_key]["column"]
chart_label = config.KPI_DEFINITIONS[metric_key]["label"]
chart_unit = config.KPI_DEFINITIONS[metric_key]["unit"]

# Build Plotly line chart
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=filtered_df[config.DATE_COLUMN],
    y=filtered_df[chart_column],
    mode="lines+markers",
    line=dict(width=2),
    marker=dict(size=4),
    name=chart_label,
    hovertemplate=(
        f"<b>{chart_label}</b>: %{{y:.1f}}{chart_unit}<br>"
        "Date: %{x|%B %Y}<extra></extra>"
    ),
))

# Add trend-break vertical line at Jan 2025
break_date = pd.Timestamp(config.POP_REF_BREAK_DATE)
if start_date <= break_date <= end_date:
    latest_y_val = filtered_df[chart_column].max()
    fig.add_vline(
        x=break_date,
        line_dash="dash",
        line_color="red",
        annotation_text="2025 pop-ref break",
        annotation_position="top left",
        annotation_font=dict(color="red", size=10),
    )

fig.update_layout(
    title=f"{chart_label} — Monthly Trend",
    xaxis_title="Month",
    yaxis_title=chart_unit,
    height=config.CHART_HEIGHT,
    hovermode="x unified",
    template="plotly_white",
)

st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Executive Insights
# ---------------------------------------------------------------------------
st.subheader("💡 Executive Insights")

insights = generate_insights(filtered_df)

for insight in insights:
    if insight.is_trend_break_warning:
        st.warning(insight.text, icon="⚠️")
    else:
        st.info(insight.text, icon="📋")


# ---------------------------------------------------------------------------
# Methodology & Limitations
# ---------------------------------------------------------------------------
with st.expander("📘 Methodology & Limitations"):
    st.markdown("### Data Source")
    st.markdown(
        f"- **Provider:** {config.SOURCE_NAME}"
        f"\n- **Dataset:** Monthly Principal Labour Force Statistics"
        f"\n- **CSV URL:** `{config.CSV_URL}`"
        f"\n- **License:** {config.LICENSE}"
        f"\n- **Collection method:** Labour Force Survey (LFS), a monthly survey"
        f" designed to collect representative data on the labour force at"
        f" national and state level."
    )

    st.markdown("### Metric Definitions")
    st.markdown(
        "| Metric | Definition | Unit |"
        "\n|---|---|---|"
        "\n| Labour Force Size | Employed + Unemployed persons | Thousands |"
        "\n| Participation Rate | Labour force ÷ Working-age population (15–64) | % |"
        "\n| Employment-to-Population Ratio | Employed ÷ Working-age population (15–64) | % |"
        "\n| Unemployment Rate | Unemployed ÷ Labour force | % |"
    )

    st.markdown("### Important Caveats")
    st.markdown(
        f"1. **Population-reference change (2025):** Data prior to 2025 is "
        f"compiled using population benchmarks from the **2010 Census**. "
        f"Data from 2025 onwards uses population benchmarks from the "
        f"**2020 Census**. This creates a methodological break around "
        f"January 2025 — observed jumps in rate KPIs during this period "
        f"may reflect the revised denominator, not necessarily economic change."
        f"\n\n2. **Rounding:** Values are rounded to one decimal place. "
        f"Sums of categories (e.g., employed + unemployed) may not always "
        f"equal the total shown due to independent rounding."
        f"\n\n3. **Data freshness:** The dashboard caches data locally. "
        f"If the official source is unavailable, the most recent cached "
        f"data is shown with a visible warning."
        f"\n\n4. **No causal claims:** This dashboard describes observed "
        f"trends only. It does not diagnose causes of labour-market changes."
    )

    st.markdown("### How Comparisons Work")
    st.markdown(
        "- **Month-on-Month (MoM):** Compares the latest month to the "
        "previous month. For rate KPIs, the change is shown in "
        "**percentage points**. For count KPIs, the change is shown as a "
        "**relative percentage**."
        "\n- **Year-on-Year (YoY):** Compares the latest month to the same "
        "month one year earlier."
        "\n- **Trend-break warnings:** When a comparison crosses January 2025, "
        "a `*trend-break*` flag is appended to alert the reader."
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
summary_text = get_data_summary_text(df, metadata)
st.caption(summary_text)



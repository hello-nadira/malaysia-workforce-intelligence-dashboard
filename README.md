# Malaysia Workforce Intelligence Dashboard

**[Click for Live Dashboard](https://malaysia-workforce-intelligence-dashboard-zp3od2bwek7daqeapt4t.streamlit.app)** | **[View Source Code on GitHub](https://github.com/hello-nadira/malaysia-workforce-intelligence-dashboard)**

An executive-ready business intelligence dashboard that visualises Malaysia's monthly labour force statistics using official open data from the Department of Statistics Malaysia (DOSM).

## Dashboard Preview

![Malaysia Workforce Intelligence Dashboard]
<img width="1402" alt="Malaysia Workforce Intelligence Dashboard" src="https://github.com/user-attachments/assets/32fd571b-5d11-44b8-b7ae-27fc4e2ad4db" />
## Overview

This project uses Python, Pandas, Streamlit and Plotly to turn monthly labour force statistics into an interactive dashboard for exploring Malaysia's employment conditions.

## Features

- **Key performance indicators:** Labour force, employment, unemployment, participation rate and employment-to-population ratio, as implemented in the dashboard.
- **Monthly and yearly comparisons:** Explore changes over time, where supported by the available data.
- **Executive insights:** Rule-based summaries of labour market indicators.
- **Data validation:** Check incoming data before using it in the dashboard.
- **Data caching and fallback:** Reuse cached data when the live data source is unavailable, with a warning when fallback data is used.
- **Methodology awareness:** Flag the population-reference change introduced in 2025 when interpreting comparisons across that period.



## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/hello-nadira/malaysia-workforce-intelligence-dashboard.git
cd malaysia-workforce-intelligence-dashboard
```

### 2. Create and activate a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
python -m streamlit run dashboard.py
```

### 5. Run the tests

```bash
python -m pytest tests/ -v
```

## Project structure

```text
dashboard_proj/
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── kpi.py
│   ├── insights.py
│   └── metadata.py
├── tests/
│   ├── __init__.py
│   ├── test_data_validation.py
│   ├── test_kpi_calculations.py
│   └── test_date_comparisons.py
│   └── test_insights.py
└── dashboard.py
```

## Data source

- **Dataset:** Monthly Principal Labour Force Statistics
- **Provider:** Department of Statistics Malaysia (DOSM)
- **Dataset catalogue:** https://data.gov.my/data-catalogue/lfs_month
- **CSV:** https://storage.dosm.gov.my/labour/lfs_month.csv
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)

## Data dictionary

| Column | Unit | Description |
|---|---|---|
| `date` | YYYY-MM-DD | Reference month; day is always 01 |
| `lf` | Thousands of people | Labour force |
| `lf_employed` | Thousands of people | Employed persons |
| `lf_unemployed` | Thousands of people | Unemployed persons |
| `lf_outside` | Thousands of people | Persons outside the labour force |
| `p_rate` | Percent | Labour force participation rate |
| `ep_ratio` | Percent | Employment-to-population ratio |
| `u_rate` | Percent | Unemployment rate |

## Important methodological caveat

The dataset uses different population-reference benchmarks before and from January 2025: the 2010 Census benchmark for earlier data and the 2020 Census benchmark for data from 2025 onwards.

Comparisons spanning this methodological break should be interpreted with care because changes may partly reflect revised population benchmarks rather than changes in labour market conditions alone.

## Architecture decisions

- **Separation of concerns:** Data processing and business logic are organised in `src/`, separate from the Streamlit interface in `dashboard.py`.
- **Disk-based caching:** Data is cached in Parquet format under `.dashboard_cache/`, with a fallback to previously cached data if a live download fails.
- **Deterministic insights:** Executive insights are generated using rule-based logic rather than LLMs or external AI APIs.
- **Live data:** KPI figures are obtained from the dataset at runtime rather than hardcoded into the application.

## Testing

The project includes a pytest test suite for data validation, KPI calculations and date comparisons.

Run the tests with:

```bash
python -m pytest tests/ -v
```

## Tech stack

- Python
- Pandas
- Streamlit
- Plotly
- Pytest


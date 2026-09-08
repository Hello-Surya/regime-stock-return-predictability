# Research Execution Guide

## Initial setup on Windows

Open PowerShell in the directory that contains `pyproject.toml`.

Verify Python 3.11:

```powershell
py -V:3.11 --version
```

Create the environment and install the project:

```powershell
py -V:3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Using `.venv\Scripts\python.exe` directly avoids PowerShell activation-policy issues.

## Run the complete test suite

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Unit and synthetic tests do not require WRDS credentials.

## Synthetic software validation

Synthetic data are used only for software validation and are not empirical research results:

```powershell
.\.venv\Scripts\python.exe scripts\run_single_stock.py --synthetic --ticker STK001
```

Outputs are written to `results\synthetic_validation\`.

## WRDS source validation

General CRSP/VIX validation:

```powershell
.\.venv\Scripts\python.exe scripts\test_wrds_connection.py
```

Compustat/CCM validation:

```powershell
.\.venv\Scripts\python.exe scripts\validate_accounting_sources.py
```

The validated production sources are `crsp.msf_v2`, `comp.funda`, `crsp.ccmxpf_lnkhist`, and `cboe_all.cboe`.

Never place a WRDS password in Python source, YAML, Git, or chat.

## Build the complete baseline predictor panel

Quick construction:

```powershell
.\.venv\Scripts\python.exe scripts\build_modeling_panel.py --quick
```

Production construction:

```powershell
.\.venv\Scripts\python.exe scripts\build_modeling_panel.py --production
```

The quick and production modes use identical accounting, CCM, timing, unit, and B/M definitions. Quick mode only reduces computational scale.

The production panel is stored locally at:

`data\processed\modeling_panel.parquet`

and remains ignored by Git because it contains licensed row-level WRDS-derived data.

Aggregate validation outputs are written to:

`results\data_validation\production\`.

## Earlier preliminary empirical runs

The earlier two-predictor single-stock validation remains reproducible with:

```powershell
.\.venv\Scripts\python.exe scripts\run_single_stock.py --ticker AAPL --quick
```

The earlier two-predictor cross-sectional portfolio evaluation remains reproducible with:

```powershell
.\.venv\Scripts\python.exe scripts\make_portfolios.py --quick
```

Those empirical outputs use size and momentum only. They predate the completed Compustat/CCM book-to-market construction and must remain labeled as preliminary historical results.

## Current research stage

The complete production data panel now contains size, book-to-market, and momentum 12–2 with explicit leakage-safe accounting timing.

The next stage is full baseline Elastic Net and XGBoost estimation on the validated three-predictor panel under the frozen time-respecting OOS design. Formal HAC/Newey-West inference, turnover measurement, and the frozen 0/50-basis-point one-way transaction-cost robustness analysis remain subsequent tasks.

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

## Run the test suite

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The validated cross-sectional branch currently contains **38 software tests**. Unit tests do not require WRDS credentials.

## Synthetic software validation

This mode uses generated data only and is not an empirical research result:

```powershell
.\.venv\Scripts\python.exe scripts\run_single_stock.py --synthetic --ticker STK001
```

Outputs are written to `results\synthetic_validation\`.

## WRDS connection validation

```powershell
.\.venv\Scripts\python.exe scripts\test_wrds_connection.py
```

The script securely authenticates through the normal WRDS Python workflow, resolves the current CRSP monthly source, pulls a tiny common-stock sample, discovers the Cboe VIX table, pulls a tiny VIX sample, and closes the connection cleanly.

Never place a WRDS password in Python source, YAML, Git, or chat.

## Preliminary single-stock validation

The earlier real-data validation can still be reproduced with:

```powershell
.\.venv\Scripts\python.exe scripts\run_single_stock.py --ticker AAPL --quick
```

Its outputs are written to `results\preliminary\`.

## Current cross-sectional empirical run

The current empirical stage is the fixed-universe cross-sectional portfolio evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\make_portfolios.py --quick
```

The quick empirical configuration uses CRSP CIZ monthly common stocks from 2012 through 2025; a fixed top-300 eligible common-stock universe formed on 2012-01-31; a $5 feature-month price screen; size and momentum 12-2 predictors; next-calendar-month returns; expanding-median VIX regimes; historical forward-chaining model selection; and model refits every three OOS months.

The cross-sectional run persists the complete OOS prediction panel, applies explicit training-feature and realized-target timing checks, validates prediction dispersion before portfolio sorting, constructs D1-D10 equal- and lagged-market-equity value-weighted portfolios, computes D10-minus-D1 returns overall and by VIX regime, and reports monthly Spearman rank diagnostics.

Outputs are written to `results\preliminary_cross_section\`. Generated stock-level prediction and assignment files are intentionally ignored by Git and should not be committed to a public repository.

## Current research limitation

The current stage remains preliminary and uses size and momentum only. The production design still requires Compustat fundamentals, CCM linking, properly timed book equity/book-to-market, the production sample specification, formal HAC/Newey-West inference, turnover measurement, and the frozen 0/50-basis-point one-way transaction-cost robustness analysis.

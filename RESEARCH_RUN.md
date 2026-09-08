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

The current implementation contains **23 software tests**. Unit tests do not require WRDS credentials.

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

## Current empirical single-stock run

For the current real empirical stage, run:

```powershell
.\.venv\Scripts\python.exe scripts\run_single_stock.py --ticker AAPL --quick
```

The quick empirical configuration uses CRSP CIZ monthly common stocks from 2012 through 2025; a fixed top-300 eligible common-stock universe formed on 2012-01-31 with AAPL retained explicitly; complete monthly histories; a $5 feature-month price screen; size and momentum 12-2 predictors; next-calendar-month returns; expanding-median VIX regimes; historical forward-chaining model selection; and model refits every three OOS months.

Real outputs are written to `results\preliminary\`.

## Current research limitation

The current stage is deliberately a preliminary empirical validation with size and momentum only. The production design still requires Compustat fundamentals, CCM linking, properly timed book equity/book-to-market, the full CRSP universe, portfolio construction, transaction costs, and formal inference.

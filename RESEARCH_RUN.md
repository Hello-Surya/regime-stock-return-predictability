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

Milestone 3 contains **23 software tests**. Unit tests do not require WRDS credentials.

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

## Preliminary empirical single-stock run

For the current real research milestone, run:

```powershell
.\.venv\Scripts\python.exe scripts\run_single_stock.py --ticker AAPL --quick
```

The quick empirical configuration does the following:

- CRSP CIZ monthly common stocks from 2012 through 2025;
- fixed reduced universe formed from the top 300 eligible stocks by market equity on 2012-01-31, plus the selected ticker;
- complete monthly histories are pulled for that fixed universe;
- the $5 price screen is applied at feature month `t`, after target construction, so a future price decline cannot remove the `t+1` realized return;
- predictors: log market equity and momentum 12-2;
- target: next-calendar-month stock return;
- daily Cboe VIX history from 1990 onward, converted to the final observed VIX close each month;
- HIGH/LOW regime from the expanding VIX median through month `t` only;
- hyperparameters selected once using forward-chaining validation entirely before the first OOS month;
- strict expanding historical training information;
- models refitted every three OOS months as a documented quick-run computation reduction;
- final displayed output for AAPL, while the models are trained cross-sectionally.

The first live run queries WRDS and caches the resulting datasets in `data\raw\`. Subsequent runs reuse those Parquet files unless `--refresh` is supplied.

To intentionally refresh the source data:

```powershell
.\.venv\Scripts\python.exe scripts\run_single_stock.py --ticker AAPL --quick --refresh
```

## Preliminary result files

Real outputs are written to `results\preliminary\`:

- `single_stock_predictions.csv`
- `single_stock_predictions_long.csv`
- `single_stock_metrics.csv`
- `regime_metrics.csv`
- `model_comparison.csv`
- `model_selection.csv`
- `data_validation.csv`
- `run_metadata.json`
- `actual_vs_predicted.png`
- `vix_regimes.png`
- `regime_performance.png`
- `preliminary_results_summary.md`

`single_stock_predictions.csv` contains the selected stock's date, VIX, regime, realized next-month return, historical-mean benchmark, Elastic Net prediction, and XGBoost prediction.

## OOS R-squared

The reported OOS R-squared is measured against a stock-specific expanding historical-mean return forecast calculated using only information available before each prediction month. This is distinct from ordinary in-sample R-squared and from an arbitrary zero-return benchmark.

## Current research limitation

Milestone 3 is deliberately a preliminary empirical validation with size and momentum only. The production design still requires Compustat fundamentals, CCM linking, properly timed book equity/book-to-market, the full CRSP universe, and the later portfolio/inference stages.

## Common Windows / WRDS issues

- If `py` cannot find 3.11, run `py install 3.11`, then verify with `py -V:3.11 --version`.
- If pip says no `pyproject.toml` exists, navigate to the extracted repository directory that actually contains `pyproject.toml`.
- If VS Code uses a different interpreter, select `.venv\Scripts\python.exe`.
- If WRDS authentication fails, use your institution's normal WRDS authentication setup; do not hard-code credentials.
- If CRSP discovery reports no supported source, confirm that the account has CRSP Stock access.
- VIX extraction prefers the WRDS Cboe product under `cboe_all` and inspects every readable table rather than assuming a physical table name. If WRDS table resolution still fails, the empirical pipeline automatically falls back to Cboe's official VIX historical CSV (1990-present), records that provenance in the cached data, and continues.
- If both WRDS VIX discovery and the official Cboe download fail, run `scripts\test_wrds_connection.py` and share only the diagnostic table/schema message, never credentials.

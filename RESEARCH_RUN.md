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

## Full baseline model estimation

The baseline estimator reuses the completed production panel. It does not rebuild CRSP, Compustat, CCM, book-to-market, or VIX data.

Software validation only:

```powershell
.\.venv\Scripts\python.exe scripts\run_baseline_models.py --validate
```

This mode uses synthetic data only and must not be interpreted as empirical evidence.

Production OOS estimation:

```powershell
.\.venv\Scripts\python.exe scripts\run_baseline_models.py --production
```

The production runner validates the frozen completed-data counts before tuning or fitting. It uses exactly `log_me`, `book_to_market`, and `mom_12_2`; selects hyperparameters once using pre-OOS historical forward-chaining validation; freezes those parameters; and refits Elastic Net and XGBoost monthly on the expanding historical sample.

Generated row-level predictions, checkpoints, tables, figures, and metadata are written locally under:

`results\baseline_models\`

The generated directory is ignored by Git except for `.gitkeep`. Use `--refresh` only when intentionally clearing and recomputing model-selection and prediction checkpoints.

## Economic-value and regime inference

After production predictions exist, run:

```powershell
.\.venv\Scripts\python.exe scripts\run_economic_value.py
```

The default specification applies a 50-basis-point one-way transaction cost per dollar of turnover and six monthly HAC lags. Both can be changed explicitly with command-line arguments, but the default values are the frozen baseline robustness specification.

This stage reuses `baseline_oos_predictions.parquet`; it does not retrain Elastic Net or XGBoost.

It writes:

- `portfolio_monthly_returns.csv`;
- `portfolio_performance.csv`;
- `portfolio_regime_tests.csv`;
- `rank_ic_inference.csv`; and
- `rank_ic_regime_tests.csv`.

The portfolio stage uses tie-safe monthly prediction ranks, equal-weighted and lagged-market-equity value-weighted D10-minus-D1 portfolios, return-drifted turnover, and Newey–West/HAC inference.

## Earlier preliminary empirical runs

The earlier two-predictor single-stock and reduced-cross-section runs remain reproducible for historical comparison:

```powershell
.\.venv\Scripts\python.exe scripts\run_single_stock.py --ticker AAPL --quick
```

```powershell
.\.venv\Scripts\python.exe scripts\make_portfolios.py --quick
```

Those outputs use size and momentum only and are superseded by the three-predictor production baseline for headline empirical conclusions.

## Current research stage

The production three-predictor baseline and economic-value evaluation are complete. The OOS sample contains 893,486 stock-month predictions across 311 formation months from January 2000 through November 2025.

The headline production evidence is concentrated in LOW-VIX months: XGBoost LOW-VIX mean Spearman IC is 0.01867 (`p = 0.013`), and the equal-weighted XGBoost LOW-VIX D10-minus-D1 portfolio earns 0.748% per month net of the frozen 50-basis-point turnover cost specification (`p = 0.0049`). The direct LOW-minus-HIGH rank and portfolio differences are positive but not conventionally significant.

See `docs\baseline_model_estimation.md` and the paper sections for the exact methodology and interpretation.

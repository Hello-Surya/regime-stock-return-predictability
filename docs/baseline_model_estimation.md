# Production Baseline Model Estimation

## Purpose

This stage estimates the frozen three-predictor baseline specification on the completed 1990–2025 CRSP–Compustat–CCM panel. The models are Elastic Net and XGBoost. The predictors are exactly `log_me`, `book_to_market`, and `mom_12_2`; the target is the next-calendar-month stock return.

The estimator reuses `data/processed/modeling_panel.parquet`. It does not reconstruct Compustat, CCM, book equity, book-to-market, CRSP features, or VIX regimes.

## Production sample guard

Before model selection or estimation, `scripts/run_baseline_models.py --production` validates the completed-data snapshot:

- 2,140,091 total stock-months;
- 1,798,891 observations with all three baseline predictors;
- 1,340,823 observations in the final screened predictor/target/regime sample; and
- zero duplicate `permno`/`date` rows.

A material mismatch stops estimation rather than silently using a different sample.

The final baseline estimation sample is complete-case for all three predictors, the next-month target, and the VIX regime. Both models therefore receive the same effective information set. The model pipelines retain their historical preprocessing components, but the production complete-case screen means no predictor imputation is used to expand one model's sample relative to the other.

## Information timing

Formation is at month-end `t`. The saved target must be realized in the next actual calendar month, `t+1`; a next available row after a calendar gap is not accepted as the target.

For an OOS formation month `t`:

- every training feature date is strictly earlier than `t`;
- every training target realization date is on or before `t` under the repository's established month-end information convention;
- preprocessing is fitted on the historical training sample only; and
- the VIX regime attached to `t` is the regime used to evaluate the return realized at `t+1`.

When the completed panel exposes `vix_expanding_median`, the production loader audits the saved label against the frozen rule: HIGH when `VIX_t` is above the expanding median using VIX observations through `t`, otherwise LOW. The estimator never reclassifies regimes after seeing returns.

## Hyperparameter selection

Hyperparameters are selected once using only the pre-OOS history and are then frozen for the production OOS run. Validation is forward-chaining by calendar month; random or shuffled K-fold validation is not used.

The configured grids remain the baseline grids in `configs/production.yaml`. To make historical selection feasible on a laptop, parameter selection uses a deterministic cross-sectional cap of 500 observations per historical month. This sampling is confined to hyperparameter selection. Every production model refit uses the full eligible expanding historical sample.

Elastic Net standardization is learned inside its historical training pipeline. XGBoost uses `reg:squarederror`, a fixed seed, histogram tree construction, and the configured four-thread production limit.

## Model refitting and OOS predictions

The production configuration retains monthly model refits. For each OOS month, both models are estimated on all historically available baseline observations and predict the full eligible cross-section at `t`.

The primary row-level artifact is:

`results/baseline_models/baseline_oos_predictions.parquet`

It contains identifiers, formation and realization timing, the three predictors, actual next-month return, VIX/regime fields, the benchmark prediction, both model predictions, and training-end audit dates.

The output directory is ignored by Git except for `.gitkeep`; licensed or row-level WRDS-derived artifacts must remain local.

## Headline OOS R-squared benchmark

The full-panel headline OOS R-squared uses the **pooled expanding historical mean**. At formation month `t`, every stock receives the pooled mean of all next-month stock returns whose target realization dates are observable on or before `t`.

Formally, if the historical information set at `t` is `H_t`, the benchmark is

`mean(y_j,s+1 for observations in H_t)`.

The headline statistic is

`1 - SSE_model / SSE_benchmark`.

This pooled benchmark is the canonical full cross-sectional baseline. It is distinct from the stock-specific historical-mean benchmark used in the earlier single-stock AAPL validation. The change is documented because the production estimand is a pooled cross-sectional forecasting problem rather than a single-stock forecasting exercise. The benchmark is fixed ex ante and is never selected based on final model performance.

## Predictive diagnostics

The production run writes:

- `baseline_model_metrics.csv`: N, MSE, RMSE, MAE, OOS R-squared, correlation, realized-return volatility, normalized RMSE, and benchmark RMSE for OVERALL, HIGH, and LOW states;
- `baseline_model_comparison.csv`: concise paper-oriented model comparison;
- `baseline_rank_metrics.csv`: summary of monthly cross-sectional Spearman rank IC, including mean, median, standard deviation, positive-month fraction, and number of months;
- `monthly_rank_ic.csv`: month-level Spearman and Pearson IC diagnostics;
- `prediction_diagnostics.csv`: prediction distribution, coverage, extremes, and realized-return distribution diagnostics;
- `model_selection.csv`: historical CV grid results, selected parameters, and tuning interval; and
- `run_metadata.json`: sample, provenance, configuration, benchmark, selected parameters, refit schedule, seed, and configuration fingerprint.

Interpretability outputs are descriptive only. Elastic Net standardized coefficients are stored by refit date. Basic XGBoost feature importance is stored by refit date; it is not treated as structural importance or formal economic inference.

## Figures

The run produces:

- `baseline_oos_performance.png`: 12-month rolling cross-sectional forecast RMSE;
- `regime_model_performance.png`: HIGH/LOW normalized forecast error;
- `monthly_rank_ic.png`: monthly Spearman rank IC; and
- `prediction_distribution.png`: prediction distributions with display-only 0.1% tail clipping clearly labeled on the axis.

Metric calculations always use the unmodified predictions.

## Checkpointing and restart safety

Monthly prediction checkpoints are written under `results/baseline_models/checkpoints/`. The checkpoint manifest includes a fingerprint of the production panel, configuration, selected parameters, and production-estimation implementation. A mismatched fingerprint stops reuse. `--refresh` explicitly clears model-selection and monthly prediction checkpoints.

## Commands

Software validation only:

```powershell
.\.venv\Scripts\python.exe scripts\run_baseline_models.py --validate
```

Full production estimation:

```powershell
.\.venv\Scripts\python.exe scripts\run_baseline_models.py --production
```

`--validate` uses synthetic data and is not an empirical result. The manuscript and headline README empirical findings should be updated only after the full production run has completed and its outputs have been inspected.

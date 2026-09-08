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

The final production OOS prediction panel contains 893,486 stock-month observations from January 2000 through November 2025.

The baseline estimation sample is complete-case for all three predictors, the next-month target, and the VIX regime. Both models therefore receive the same effective information set.

## Information timing

Formation is at month-end `t`. The saved target must be realized in the next actual calendar month, `t+1`; a next available row after a calendar gap is not accepted as the target.

For an OOS formation month `t`:

- every training feature date is strictly earlier than `t`;
- every training target realization date is on or before `t`;
- preprocessing is fitted on the historical training sample only; and
- the VIX regime attached to `t` is the regime used to evaluate the return realized at `t+1`.

When the completed panel exposes `vix_expanding_median`, the production loader audits the saved label against the frozen rule: HIGH when `VIX_t` is above the expanding median using VIX observations through `t`, otherwise LOW.

## Hyperparameter selection

Hyperparameters are selected once using only the 1990–1999 pre-OOS history and are then frozen for the production OOS run. Validation is forward-chaining by calendar month; random or shuffled K-fold validation is not used.

Parameter selection uses a deterministic cross-sectional cap of 500 observations per historical month. This sampling is confined to hyperparameter selection. Every production model refit uses the full eligible expanding historical sample.

The selected production specifications are:

- Elastic Net: `alpha = 0.01`, `l1_ratio = 0.1`, `max_iter = 20000`;
- XGBoost: `n_estimators = 200`, `max_depth = 2`, `learning_rate = 0.01`, `subsample = 1.0`, `colsample_bytree = 0.7`, `tree_method = hist`, and `n_jobs = 4`.

## Model refitting and OOS predictions

The production configuration retains monthly model refits. For each OOS month, both models are estimated on all historically available baseline observations and predict the full eligible cross-section at `t`.

The primary row-level artifact is:

`results/baseline_models/baseline_oos_predictions.parquet`

The production run generated 893,486 OOS stock-month predictions across 311 formation months: 138 HIGH-VIX and 173 LOW-VIX months.

The output directory is ignored by Git except for `.gitkeep`; licensed or row-level WRDS-derived artifacts remain local.

## Headline OOS R-squared benchmark

The production headline OOS R-squared uses the **pooled expanding historical mean**. At formation month `t`, every stock receives the pooled mean of all next-month stock returns whose target realization dates are observable on or before `t`.

The statistic is:

`1 - SSE_model / SSE_benchmark`.

This pooled benchmark is distinct from the stock-specific historical-mean benchmark used in the earlier AAPL validation and is fixed ex ante.

## Production predictive results

Overall return-level OOS R-squared is close to zero:

| Model | OVERALL | HIGH | LOW |
| --- | ---: | ---: | ---: |
| Elastic Net | -0.000265 | -0.000680 | 0.000383 |
| XGBoost | -0.000231 | -0.001058 | 0.001063 |

Monthly cross-sectional rank IC provides a clearer regime pattern. XGBoost LOW-VIX mean Spearman IC is 0.018672 with HAC `t = 2.479` and `p = 0.0132`. XGBoost HIGH-VIX mean IC is -0.001778 with `p = 0.8866`. The direct XGBoost LOW-minus-HIGH IC difference is 0.020450 with `p = 0.1046`, so the regime contrast is suggestive but not conventionally significant.

Elastic Net rank IC is not statistically significant overall or within either regime.

## Economic-value stage

`scripts/run_economic_value.py` reuses the saved production predictions and constructs tie-safe monthly D10-minus-D1 portfolios.

The baseline portfolio design uses:

- equal weighting and lagged-market-equity value weighting;
- D10 long and D1 short;
- return-drifted pre-rebalance weights for turnover;
- 50 basis points one way per dollar of turnover; and
- Newey–West/HAC inference with six monthly lags.

The strongest baseline result is XGBoost in LOW-VIX equal-weighted portfolios:

- gross monthly D10-minus-D1 mean: 0.009690;
- gross HAC `t = 3.605`;
- net monthly mean after 50 bps turnover costs: 0.007482;
- annualized arithmetic net mean: 0.089781;
- net Sharpe ratio: 0.796;
- net HAC `t = 2.815`, `p = 0.00487`.

The corresponding value-weighted LOW-VIX XGBoost portfolio is negative after costs. The signal is therefore concentrated away from the largest stocks.

The direct LOW-minus-HIGH XGBoost equal-weighted net-return difference is 0.008458 per month with HAC `t = 1.521` and `p = 0.1282`. The paper therefore does not claim a conventionally significant direct regime difference.

## Generated research artifacts

Baseline estimation writes:

- `baseline_oos_predictions.parquet`;
- `baseline_model_metrics.csv`;
- `baseline_model_comparison.csv`;
- `baseline_rank_metrics.csv`;
- `monthly_rank_ic.csv`;
- `prediction_diagnostics.csv`;
- `model_selection.csv`;
- `run_metadata.json`;
- `elastic_net_coefficients.csv`; and
- `xgboost_feature_importance.csv`.

Economic-value inference writes:

- `portfolio_monthly_returns.csv`;
- `portfolio_performance.csv`;
- `portfolio_regime_tests.csv`;
- `rank_ic_inference.csv`; and
- `rank_ic_regime_tests.csv`.

Interpretability outputs are descriptive only. Elastic Net standardized coefficients and XGBoost feature importance are not treated as structural causal quantities.

## Figures

The production run produces:

- `baseline_oos_performance.png`;
- `regime_model_performance.png`;
- `monthly_rank_ic.png`; and
- `prediction_distribution.png`.

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

Economic value and HAC inference:

```powershell
.\.venv\Scripts\python.exe scripts\run_economic_value.py
```

`--validate` uses synthetic data and is not an empirical result.

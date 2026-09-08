# Milestone 3 — Single-Stock Out-of-Sample Validation

## Purpose

Milestone 3 establishes a genuine WRDS-based empirical path from raw monthly CRSP observations to strictly out-of-sample Elastic Net and XGBoost predictions for one selected common stock, while retaining cross-sectional model training.

## Preliminary predictor set

The initial empirical run uses:

- log market equity (size);
- momentum 12-2.

Book-to-market is intentionally omitted until Compustat/CCM accounting-data construction is implemented. This avoids substituting an invalid or contemporaneous accounting proxy for the intended production characteristic.

## Reduced-universe design

Quick mode forms a fixed reduced universe using the top 300 eligible common stocks by market equity on 2012-01-31, with the selected ticker retained. It then pulls complete monthly histories for those PERMNOs through 2025.

The universe is fixed using start-of-sample information rather than re-selected using future months. This prevents future cross-sectional membership from determining whether historical or next-month observations exist in the modeling sample.

## Timing protections

- next-month return is defined by the next **calendar month**, not merely the next available row;
- momentum 12-2 requires the eleven calendar-month returns from `t-12` through `t-2`;
- calendar gaps remain missing rather than being collapsed;
- the $5 price screen is applied at feature month `t` after target construction;
- monthly VIX uses the final daily Cboe VIX observation in the month;
- the expanding VIX median uses observations through `t` only;
- time-CV training dates strictly precede validation dates;
- OOS training information strictly precedes each prediction month;
- all imputation/scaling is fitted inside the relevant historical training sample.

## Hyperparameter selection

Quick mode selects a computationally small Elastic Net and XGBoost grid using forward-chaining validation inside the initial pre-OOS historical sample. The selected parameters are frozen for the preliminary experiment and saved to `model_selection.csv` and `run_metadata.json`.

## Evaluation

For the selected ticker, metrics are reported overall and separately in HIGH and LOW VIX regimes:

- MSE;
- RMSE;
- MAE;
- conventional realized-vs-predicted R-squared;
- OOS R-squared versus an expanding stock-specific historical-mean return forecast;
- prediction-realization correlation.

No statistical-significance claim is made in this milestone.
## Verified preliminary empirical findings

The first completed real-data run uses AAPL as the displayed stock and produces 83 OOS months from 2019-01-31 through 2025-11-30. The HIGH/LOW regime split is 42/41 months.

- Elastic Net overall: RMSE 0.08096, MAE 0.06873, OOS R-squared -0.0274.
- XGBoost overall: RMSE 0.08052, MAE 0.06833, OOS R-squared -0.0162.
- HIGH-VIX OOS R-squared: -0.0132 (Elastic Net), -0.0067 (XGBoost).
- LOW-VIX OOS R-squared: -0.0601 (Elastic Net), -0.0384 (XGBoost).

The historical-mean benchmark therefore outperforms both models in the preliminary AAPL time-series validation, while XGBoost is modestly better than Elastic Net on forecast-error metrics. Benchmark-relative performance is less negative in HIGH-VIX months than LOW-VIX months. No formal significance or economic-value claim is made from this single-stock validation; the next research stage evaluates cross-sectional ranking through predicted-return portfolio sorts.


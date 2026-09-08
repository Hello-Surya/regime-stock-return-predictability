# Single-Stock Out-of-Sample Validation

## Current stage

The project currently supports a genuine WRDS-based empirical path from monthly CRSP observations to strictly out-of-sample Elastic Net and XGBoost predictions for one selected common stock while retaining cross-sectional model training.

The preliminary predictor set contains log market equity and momentum 12-2. Book-to-market is intentionally omitted until Compustat/CCM accounting-data construction is implemented.

Quick mode forms a fixed reduced universe using the top 300 eligible common stocks by market equity on 2012-01-31, retains the selected ticker, and pulls complete monthly histories through 2025. The design includes calendar-safe next-month targets, calendar-safe momentum, a feature-month $5 price screen, expanding-median VIX regimes, forward-chaining model selection, and strict historical OOS training.

## Verified preliminary empirical findings

The completed AAPL run contains 83 OOS months from 2019-01-31 through 2025-11-30, split into 42 HIGH-VIX and 41 LOW-VIX months.

- Elastic Net overall RMSE: 0.08096; OOS R-squared: -0.0274.
- XGBoost overall RMSE: 0.08052; OOS R-squared: -0.0162.
- HIGH-VIX OOS R-squared: -0.0132 (Elastic Net), -0.0067 (XGBoost).
- LOW-VIX OOS R-squared: -0.0601 (Elastic Net), -0.0384 (XGBoost).

The historical-mean benchmark outperforms both models in this AAPL time-series validation, while XGBoost is modestly better than Elastic Net on forecast-error metrics. The next research stage evaluates cross-sectional ranking through predicted-return portfolio sorts.

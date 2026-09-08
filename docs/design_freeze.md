# Baseline Design Freeze

The baseline predicts next-month common-stock returns using characteristics observed at month t. Volatility regimes use VIX at t relative to the expanding historical median through t. OOS training uses only feature dates strictly before each prediction month. Baseline models are Elastic Net and XGBoost.

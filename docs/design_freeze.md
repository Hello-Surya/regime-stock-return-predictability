# Baseline Design Freeze

The baseline predicts next-month U.S. common-stock returns using characteristics observable at formation month t.

## Predictors

The frozen baseline predictor set is:

1. log market equity;
2. book-to-market; and
3. momentum 12–2.

No additional stock characteristics are introduced in the baseline data-construction stage.

## Target

The prediction target at formation month t is the stock return in the next actual calendar month, t+1. Calendar gaps are not treated as consecutive observations.

## Accounting timing

Book-to-market uses annual Compustat fundamentals linked to CRSP through CCM.

For a fiscal-year end in calendar year y-1, the accounting characteristic first becomes eligible in June y and remains active through May y+1. Book equity is paired with firm-level CRSP market equity from December y-1. Accounting information is not immediately forward-filled from `datadate`.

## Volatility regime

The regime at month t uses VIX at t relative to the expanding historical median calculated through t. Future VIX observations do not enter the regime label.

## Out-of-sample design

Elastic Net and XGBoost are the frozen baseline forecasting models. OOS training uses only feature dates strictly before each prediction month, and transformations or model selection must not use future test observations.

The earlier empirical results in the repository were produced with the preliminary two-predictor specification containing size and momentum only. They remain historical preliminary results and are not relabeled as three-predictor estimates.

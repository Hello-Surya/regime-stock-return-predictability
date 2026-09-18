# Baseline Modeling Panel Validation

## Sample

- Formation period: 1990-01-31 through 2025-12-31
- Realized-return period: 1990-02-28 through 2026-01-31
- Observations: 2,140,091
- Securities: 18,898
- Complete baseline observations: 1,340,823

## Baseline predictors

- Size: log_me
- Book-to-market: book_to_market
- Momentum 12-2: mom_12_2
- Target: next_month_return

## Predictor coverage

- log_me: 99.96%
- book_to_market: 84.95%
- mom_12_2: 92.16%

## Accounting timing

Accounting lineage is retained and validation rejects accounting data used before its June eligibility date.

## Target timing

Every realized-return date is exactly the next calendar month-end after formation, and training eligibility is governed by target observability.

## Volatility regimes

- HIGH months: 193
- LOW months: 239
- Timing check: historical_reference_verified

## Duplicate-key check

Duplicate permno + formation_date rows: 0

## Monthly cross-sectional coverage

- Minimum complete-case securities: 0
- Median complete-case securities: 2737.0
- Maximum complete-case securities: 4915

## Data-quality observations

Raw economically defined predictors are retained. This stage applies neither full-sample standardization nor implicit winsorization.

## Current limitations

Row-level WRDS-derived panel and extreme-observation audit files remain local and are excluded from public version control.

## Next Stage

Full baseline out-of-sample model estimation uses this validated canonical panel with Elastic Net and XGBoost.

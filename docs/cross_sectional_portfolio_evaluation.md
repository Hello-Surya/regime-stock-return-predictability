# Cross-Sectional Portfolio Evaluation

This stage evaluates whether out-of-sample Elastic Net and XGBoost forecasts contain useful cross-sectional ranking information in the preliminary two-predictor specification (`log_me` and `mom_12_2`). Book-to-market remains deferred until Compustat/CCM construction.

## Timing

At formation month `t`, model inputs and the VIX regime use information available through `t`. The realized portfolio return is the canonical next-calendar-month CRSP return at `t+1`. The prediction panel records `formation_date`, `realized_return_date`, `train_feature_end_date`, and `train_target_end_date`. The OOS engine requires the fitted model's latest feature date to be strictly before formation and its latest realized training-target date to be no later than formation.

## Fixed research universe

Quick mode uses the configured top-300 formation-month research universe. The existing single-stock extraction can retain one selected ticker in addition to that top-N set. Cross-sectional evaluation therefore reconstructs the top-N formation set from cached formation-month market equity and PERMNO, removing any security present only because of selected-ticker retention. Monthly eligibility is then applied within that fixed universe.

## Value weights

`me_lag` is the previous calendar month's market equity. Feature construction maps it using a calendar-complete monthly series, so a calendar gap produces a missing weight rather than an artificial lag. `me_lag_date`/`value_weight_date` records the timing explicitly. The portfolio code never creates an additional lag from `me_lag`. Value-weighted decile returns use only finite positive weights and never fall back to equal weighting when weights are invalid.

## Decile assignment and degeneracy safeguards

For each formation month and model independently, valid predictions are sorted from lowest to highest. Ordinary exact prediction ties are broken deterministically by PERMNO. Ordinal ranks are mapped into ten approximately equal-sized portfolios using `floor((rank - 1) * 10 / N) + 1`.

A month-model group is explicitly skipped when it has fewer than ten valid predictions, fewer than ten distinct prediction values, or a numerically negligible prediction range. This prevents security-identifier tie-breaking from manufacturing D1-D10 portfolios when a fitted model has effectively collapsed to an intercept-only forecast.

D1 contains the lowest predicted returns and D10 the highest. The monthly long-short return is `D10 - D1`.

## Reported diagnostics

The stage produces equal- and value-weighted D1-D10 returns, monthly D10-D1 series, overall/HIGH/LOW descriptive summaries, and monthly Spearman cross-sectional rank correlations. OVERALL statistics are computed directly from the complete monthly series rather than by averaging regime summaries. Annualized mean is `12 * monthly mean`, annualized volatility is `sqrt(12) * monthly sample standard deviation`, and the descriptive Sharpe uses a zero risk-free rate. Formal HAC/Newey-West inference is intentionally deferred.

Monthly Spearman correlation is reported only when both predictions and realized returns contain enough variation for the statistic to be defined. Degenerate prediction cross-sections are recorded rather than passed through correlation routines as warning-generating pseudo-results.

Cumulative long-short figures use the cumulative arithmetic sum of monthly D10-D1 returns and are not labeled as wealth indices.

## Validated preliminary findings

The corrected real-data run covers 83 OOS formation months from January 2019 through November 2025. Monthly eligible-stock counts range from 225 to 251, with a median of 240. All timing checks pass, accepted portfolio months contain complete D1-D10 portfolios, and all accepted EW/VW long-short returns are finite.

The tuned Elastic Net specification generates exactly one distinct forecast in every OOS month. All 83 Elastic Net month-model groups are therefore excluded from portfolio formation as `fewer_than_10_distinct_predictions`. This is recorded as a substantive null cross-sectional result for the current size-and-momentum specification.

XGBoost retains valid forecast dispersion in all 83 OOS months. Its mean monthly Spearman rank correlation is -0.0101 overall, -0.0115 in HIGH-VIX months, and -0.0086 in LOW-VIX months. The equal-weighted D10-D1 spread is approximately -0.017% per month overall. The value-weighted spread is approximately 0.708% per month overall and 1.432% per month in LOW-VIX months, while the HIGH-VIX value-weighted spread is essentially zero.

These results are descriptive and preliminary. They do not yet include formal HAC inference, turnover, transaction costs, or the complete Compustat/CCM predictor set.

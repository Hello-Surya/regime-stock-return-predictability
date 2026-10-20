# Turnover and Transaction-Cost Analysis

This module is a **production validation-gated implementation**, not a report of empirical findings. Only the user’s local research machine has the ignored row-level inputs needed for actual production results.

## Frozen inputs

- `results/economic_value/portfolio_assignments.csv` is the authoritative Milestone 8 stock-month decile assignment file. No new ranking is created.
- `results/economic_value/decile_returns_monthly.csv` and `results/economic_value/long_short_returns.csv` are the authoritative gross returns. Reconstructed decile returns, dates, counts, and regimes must match.
- `results/inference/long_short_hac_inference.csv` and `results/inference/long_short_regime_differences.csv` are the gross Milestone 9 benchmark for a full no-change statistical replication check.
- `results/baseline_models/baseline_oos_predictions.parquet` is only hashed and protected. It is never read for sorting or training.

All files are SHA-256 hashed before and after the execution. An input mismatch aborts rather than allowing new portfolios, changed forecasts, or new inference choices.

## Timing and weights

At month-end formation date t the saved assignment contains t+1 realized returns, used exclusively to measure t+1 gross performance and to drift holdings into the *next* portfolio formation. The pre-trade weight at t uses *only* the prior assignment at t-1 and its holding-period realized return in t:

\[
\widetilde{w}_{i,t} = \frac{w_{i,t-1}(1+R_{i,t})}{\sum_j w_{j,t-1}(1+R_{j,t})}.
\]

EW assigns 1/N; VW uses the same strictly positive and finite lagged market equity as M8. The union of previous and current holdings permits purchases, exits, and drift. A missing prior return, nonconsecutive month, or failure to normalize is a fatal error, not silently repaired.

## Turnover and costs

\[
T_{d,t}=\frac{1}{2}\sum_i |w_{i,t}-\widetilde{w}_{i,t}|.
\]

Initial cash formation is explicitly charged one full unit of one-sided turnover per leg. A basis point equals 0.0001 of portfolio capital; 50 bps equals 0.005, **not** 50%. Two scenarios are frozen: 0 bps and 50 bps one way.

\[
R^{LS,net}_{t+1} = R^{D10}_{t+1}-R^{D1}_{t+1}-c\,(T_{D10,t}+T_{D1,t}).
\]

With $1 long D10 and $1 short D1, gross exposure is $2 and net target dollar exposure is $0. The sum of two one-sided turnover measures is the traded notional normalization of this $1/$1 strategy, not turnover relative to the $2 gross exposure. Do not halve it again.

`D1_net` represents the long-only D1 portfolio after its trading costs; `D1_short_leg_net_contribution` is the realized contribution to the short-side spread. The combined net spread includes both trading costs.

## Statistical inference

The 50-bps monthly spread is passed to the same six-lag Newey--West/HAC functions used by M9. Regime mean tests are reported for overall, LOW, MIDDLE, HIGH; the three pre-specified dummy-regression contrasts are MIDDLE−LOW, HIGH−LOW and HIGH−MIDDLE. Holm correction operates independently within each model-by-weighting family; BH/FDR p-values are retained secondarily.

## Local outputs and reproducibility

The executable is `scripts/run_transaction_costs.py --production`. It writes a separate `results/transaction_costs/` directory with all-ten-decile turnover, D1/D10 leg turnover, 0-/50-bps net monthly spreads, overall/binary/three-state summaries, break-even rates, three figures, a publication table, and metadata. `paper/generated/table_transaction_costs.tex` is constructed only from real validated outputs.

`results/transaction_costs/portfolio_weights_monthly.csv` is a **local row-level audit artifact** containing D1/D10 entries and exits. It is automatically added to `.gitignore`; it must never be committed if restricted by WRDS or upstream dataset licenses.

The local run updates the Research Terminology and Methodology Appendix, the README and the manuscript only **after** all M8/M9 and arithmetic checks pass; it then executes the strict terminology audit. It never commits or pushes automatically.

## Interpretation limits

The assumed 50-bps fee is proportional, not empirically measured. Short borrow fees, rebate/financing costs and nonlinear impact are excluded. A positive net mean does not prove that the strategy is implementable. Real results must be judged against their HAC confidence intervals and regime-difference-adjusted p-values, not only the within-regime estimates.

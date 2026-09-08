# Regime-Dependent Predictability of Stock Returns

Research software for testing whether next-month U.S. common-stock return predictability differs across high- and low-volatility market regimes. The production baseline compares Elastic Net and XGBoost under a strictly time-respecting expanding-window design using size, book-to-market, and momentum 12–2.

## Current Stage of the Project

The project has completed the **production three-predictor baseline estimation and economic-value evaluation stage**.

The production data pipeline constructs a leakage-safe monthly CRSP–Compustat–CCM panel using:

- CRSP monthly stock data: `crsp.msf_v2` (CIZ);
- Compustat annual fundamentals: `comp.funda`;
- CRSP/Compustat link history: `crsp.ccmxpf_lnkhist`; and
- Cboe VIX: `cboe_all.cboe`.

The production research panel spans January 1990 through December 2025 and contains 2,140,091 stock-month observations across 18,898 PERMNOs. There are 1,798,891 observations with all three frozen baseline predictors and 1,340,823 observations in the final predictor/target/regime sample after the feature-month price and timing screens. The panel has zero duplicate `permno`/`date` rows.

The production out-of-sample evaluation covers January 2000 through November 2025, with 893,486 stock-month predictions across 311 formation months: 138 HIGH-VIX months and 173 LOW-VIX months.

## Baseline Design

The frozen predictor set is:

- `log_me` — log market equity;
- `book_to_market` — annual book-to-market with conservative June availability; and
- `mom_12_2` — momentum 12–2.

The VIX regime is observed at formation month-end `t` and is HIGH when VIX is above its expanding historical median using information through `t`; otherwise it is LOW. The target is the realized return in calendar month `t+1`.

Hyperparameters are selected once using only 1990–1999 historical data with forward-chaining validation and are frozen for the OOS period. Both models are then refit monthly on the expanding historical sample. The production OOS R-squared benchmark is the pooled expanding historical mean of returns whose realizations are observable by the formation date.

## Production Baseline Findings

Return-level forecast-error improvements over the historical-mean benchmark are very small. Overall OOS R-squared is approximately -0.00027 for Elastic Net and -0.00023 for XGBoost. In LOW-VIX months it becomes slightly positive for both models (0.00038 for Elastic Net and 0.00106 for XGBoost), whereas both models remain negative in HIGH-VIX months.

The clearer result appears in cross-sectional ranking. XGBoost has a LOW-VIX mean monthly Spearman information coefficient of 0.01867 with Newey–West/HAC `t = 2.48` and `p = 0.013`. Its HIGH-VIX mean rank IC is -0.00178 (`p = 0.887`). Elastic Net does not show statistically significant rank IC in either regime.

The direct LOW-minus-HIGH rank-IC difference is positive for XGBoost (0.02045) but is not statistically significant at conventional levels (`p = 0.105`). The evidence should therefore be described as predictive content that is **concentrated in LOW-VIX months**, not as a formally established LOW-versus-HIGH difference.

## Economic Value

Prediction-sorted decile portfolios are formed monthly using tie-safe ranks. D10 is long the highest predicted returns and D1 is short the lowest. Results are computed with equal weights and lagged-market-equity value weights. Transaction-cost robustness uses 50 basis points one way per dollar of turnover, with portfolio weights drifted through realized returns before rebalancing.

The strongest production result is the XGBoost equal-weighted LOW-VIX D10-minus-D1 portfolio:

- gross mean monthly return: 0.969%;
- gross HAC `t = 3.61`;
- net mean monthly return after 50 bps turnover costs: 0.748%;
- annualized arithmetic net mean: 8.98%;
- net Sharpe ratio: 0.80; and
- net HAC `t = 2.82`, `p = 0.0049`.

The corresponding HIGH-VIX equal-weighted XGBoost portfolio has a net mean monthly return of -0.098% and is not statistically significant.

The signal does **not** survive value weighting. XGBoost LOW-VIX value-weighted performance is economically small before costs and negative after the 50 bps turnover adjustment. This indicates that the economic signal is concentrated away from the largest stocks.

The direct LOW-minus-HIGH XGBoost equal-weighted net-return difference is 0.846% per month, but the HAC test is not conventionally significant (`t = 1.52`, `p = 0.128`). This qualification is retained in the manuscript.

## Interpretation

The production baseline does not support a claim of strong unconditional next-month return forecasting. Instead, it provides evidence that nonlinear cross-sectional predictive content is concentrated in LOW-VIX states and is economically visible in equal-weighted portfolios. The same effect is weak under value weighting, and direct LOW-versus-HIGH differences are estimated imprecisely.

These findings motivate the next research stages: temporal-stability analysis, alternative regime definitions, richer characteristic sets, and additional portfolio/transaction-cost robustness checks.

## Reproducibility

See `RESEARCH_RUN.md` for Windows/PowerShell commands.

Methodology is documented in:

- `docs/data_sources.md`;
- `docs/baseline_model_estimation.md`; and
- the production paper sections under `paper/sections/`.

Licensed WRDS-derived row-level data and generated prediction artifacts remain local and are excluded from Git.

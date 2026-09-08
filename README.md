# Regime-Dependent Predictability of Stock Returns

Research software for testing whether next-month U.S. common-stock return predictability differs across high- and low-volatility market regimes. The baseline compares Elastic Net and XGBoost under a strictly time-respecting expanding-window design.

## Current Stage of the Project

The project has reached the **preliminary real-data cross-sectional portfolio evaluation stage**.

The implemented research pipeline currently supports:

- executable synthetic software validation;
- live WRDS CRSP CIZ source discovery, extraction, normalization, caching, and validation;
- fixed-universe cross-sectional model estimation and OOS prediction;
- Cboe VIX regime construction using an expanding historical median;
- time-respecting forward-chaining hyperparameter selection;
- Elastic Net and XGBoost next-month return forecasts;
- explicit training-feature and realized-target timing audits;
- deterministic D1-D10 portfolio construction with safeguards against degenerate prediction cross-sections;
- equal- and lagged-market-equity value-weighted portfolio returns;
- overall and HIGH/LOW regime summaries;
- monthly cross-sectional Spearman rank diagnostics;
- automatically generated research tables, figures, and validation reports.

The current real empirical specification intentionally uses a **preliminary partial feature set** of size and momentum. Book-to-market is not fabricated; it will enter after the Compustat/CCM data-construction stage.

### Current Empirical Findings

The cross-sectional OOS sample contains 83 formation months from January 2019 through November 2025, with 225 to 251 eligible stocks per month after the current screens.

The tuned Elastic Net specification collapses to one identical forecast for every stock in every OOS month. Because it contains no usable within-month ranking information, all 83 Elastic Net month-groups are treated as non-sortable and no Elastic Net decile-return result is reported. This is a substantive null result for the current two-predictor regularized linear specification.

XGBoost produces valid cross-sectional forecast dispersion in all 83 OOS months. Its average monthly Spearman rank correlation is slightly negative overall (-0.0101), so the current model does not show broad monotonic ranking ability across the full cross-section.

XGBoost equal-weighted D10-minus-D1 returns are economically close to zero overall (-0.017% per month). Value-weighted results are more positive: the overall spread averages approximately 0.708% per month. This value-weighted result is concentrated in LOW-VIX months, where the mean spread is approximately 1.432% per month, while the HIGH-VIX value-weighted spread is essentially zero. These are descriptive preliminary results and are not yet accompanied by formal HAC inference or transaction-cost adjustments.

The earlier AAPL single-stock validation remains useful as a software and forecast-error check, but it is no longer the most advanced empirical stage of the project.

### Next Stage

The next stage adds the full Compustat/CCM predictor construction, formal inference, additional portfolio diagnostics, turnover, and the frozen transaction-cost robustness specification of 0 and 50 basis points one way. The current regime results are treated as preliminary and may change materially once the complete predictor set is available.

See `RESEARCH_RUN.md` for exact Windows/PowerShell commands and `docs/cross_sectional_portfolio_evaluation.md` for the cross-sectional design and validation rules.

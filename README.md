# Regime-Dependent Predictability of Stock Returns

Research software for testing whether next-month U.S. common-stock return predictability differs across high- and low-volatility market regimes. The baseline compares Elastic Net and XGBoost under a strictly time-respecting expanding-window design.

## Current Stage of the Project

The project has reached the **real single-stock out-of-sample validation stage**.

The implemented research pipeline currently supports:

- executable synthetic software validation;
- live WRDS CRSP CIZ source discovery, extraction, normalization, caching, and validation;
- real WRDS-based cross-sectional training with final OOS output for a selected common stock;
- Cboe VIX regime construction using an expanding historical median;
- time-respecting forward-chaining hyperparameter selection;
- Elastic Net and XGBoost next-month return forecasts;
- overall and HIGH/LOW volatility-regime evaluation;
- automatically generated research tables, figures, diagnostics, and a preliminary-results summary.

The current real empirical specification intentionally uses a **preliminary partial feature set** of size and momentum. Book-to-market is not fabricated; it will enter after the Compustat/CCM data-construction stage.

### Current Empirical Finding

The verified AAPL validation contains 83 OOS prediction months from January 2019 through November 2025. XGBoost modestly improves on Elastic Net in RMSE and MAE, but neither model beats the stock-specific expanding historical-mean benchmark. Both models have less-negative benchmark-relative OOS R-squared in HIGH-VIX months than in LOW-VIX months. These are preliminary validation findings, not final portfolio or inference results.

### Next Stage

The next empirical stage evaluates **cross-sectional economic value** by sorting stocks on predicted returns and constructing D10-minus-D1 portfolios overall and separately in HIGH- and LOW-VIX regimes.

See `RESEARCH_RUN.md` for exact Windows/PowerShell commands.

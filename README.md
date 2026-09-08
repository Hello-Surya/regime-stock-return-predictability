# Regime-Dependent Predictability of Stock Returns

Research software for testing whether next-month U.S. common-stock return predictability differs across high- and low-volatility regimes. The baseline compares Elastic Net and XGBoost under a strictly time-respecting expanding-window design.

Current implemented research milestones:

- **Milestone 1:** executable synthetic software-validation pipeline;
- **Milestone 2:** live WRDS CRSP CIZ discovery, extraction, normalization, caching, and validation;
- **Milestone 3:** real WRDS single-stock OOS validation from a reduced cross-sectional training universe, including Cboe VIX regime construction, historical forward-chaining hyperparameter selection, regime-specific metrics, figures, and a generated preliminary-results summary.

The first real Milestone 3 run intentionally uses a **preliminary partial feature set** of size and momentum. Book-to-market is not fabricated; it enters after the Compustat/CCM data-construction milestones.

See `RESEARCH_RUN.md` for exact Windows/PowerShell commands.

## Verified preliminary AAPL result

The first real Milestone 3 run produces 83 AAPL OOS prediction months from January 2019 through November 2025. XGBoost modestly improves on Elastic Net in RMSE and MAE, but neither model beats the stock-specific expanding historical-mean benchmark. Both models have less-negative benchmark-relative OOS R-squared in HIGH-VIX months than in LOW-VIX months. These are preliminary validation findings, not final portfolio or inference results.

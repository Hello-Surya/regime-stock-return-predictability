# Regime-Dependent Predictability of Stock Returns

Research software for testing whether next-month U.S. common-stock return predictability differs across high- and low-volatility market regimes. The baseline compares Elastic Net and XGBoost under a strictly time-respecting expanding-window design.

## Current Stage of the Project

The project has reached the **complete baseline predictor data-construction stage**.

The production data pipeline now constructs a leakage-safe monthly CRSP–Compustat–CCM panel containing the frozen baseline predictor set:

- log market equity (size);
- book-to-market; and
- momentum 12–2.

The live production build resolves and uses:

- CRSP monthly stock data: `crsp.msf_v2` (CIZ);
- Compustat annual fundamentals: `comp.funda`;
- CRSP/Compustat link history: `crsp.ccmxpf_lnkhist`; and
- Cboe VIX: `cboe_all.cboe`.

The production research panel spans January 1990 through December 2025 and contains 2,140,091 stock-month observations across 18,898 PERMNOs. It contains 1,818,006 stock-months with valid book-to-market and 1,798,891 stock-months with all three baseline predictors before the downstream price/target eligibility sequence. The final panel has zero duplicate `permno`/`date` rows.

After the sequential next-month-target, $5 feature-month price, size, momentum, CCM, book-equity, and book-to-market requirements, 1,340,823 stock-month observations remain in the complete baseline sample flow.

### Accounting construction

Book equity is formed from Compustat annual data using a documented fallback hierarchy. Stockholders' equity uses `SEQ`, then `CEQ + PSTK`, then `AT - LT`. Preferred stock uses `PSTKRV`, then `PSTKL`, then `PSTK`, with an explicit zero when all preferred-stock fields are missing. Deferred taxes and investment tax credit use `TXDITC`, then `TXDB + ITCB`, with an explicit zero when all such fields are missing.

CCM links are restricted to link types `LC` and `LU` and primary indicators `P` and `C`, and must be effective on the Compustat fiscal-year-end date. Duplicate or competing mappings are resolved deterministically before characteristics are assigned.

For a fiscal-year end in calendar year y-1, the accounting characteristic first becomes eligible in June of year y and remains active through May of year y+1. Book equity is paired with firm-level CRSP market equity from December y-1. CRSP price-times-shares market equity is in USD thousands and is converted to USD millions for the book-to-market denominator so that units match Compustat.

Nonpositive book equity, nonpositive December market equity, or an incomplete required December denominator does not produce a valid book-to-market observation.

### Production validation

The completed production build reports:

- duplicate `permno`/`date` rows: 0;
- valid size observations: 2,139,259;
- valid momentum observations: 1,972,307;
- valid book-to-market observations: 1,818,006;
- all-three-predictor observations: 1,798,891;
- median book-to-market: approximately 0.558;
- 99.9th percentile book-to-market: approximately 11.60; and
- book-to-market unit sanity check: pass.

No winsorization is introduced by the data-construction pipeline.

### Earlier Preliminary Empirical Results

The existing single-stock and cross-sectional portfolio results remain **preliminary historical results from the earlier two-predictor specification** using size and momentum only.

Those results must not be interpreted as estimates from the newly completed three-predictor panel. In particular, the previously reported Elastic Net forecast degeneracy and XGBoost portfolio diagnostics were generated before book-to-market was added.

## Next Stage

The next research stage is **full baseline model estimation using the validated three-predictor production panel**.

That stage will rerun the frozen Elastic Net and XGBoost specifications with the complete predictor set under the existing time-respecting out-of-sample design. Formal portfolio inference, turnover, and the frozen transaction-cost robustness specification remain subsequent research tasks.

See `RESEARCH_RUN.md` for Windows/PowerShell commands and `docs/data_sources.md` for the complete data-construction methodology.

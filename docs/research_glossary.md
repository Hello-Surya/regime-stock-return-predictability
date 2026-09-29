# Research Terminology and Methodology Appendix

## Purpose

This document is the canonical technical dictionary for the **Regime-Dependent Predictability of Stock Returns** repository. It defines terminology used in the code, configuration, validation outputs, figures, README, and manuscript, and records the project's exact interpretation of terms whose generic textbook meaning is not sufficiently precise.

The reference is synchronized to the current production baseline: a monthly CRSP--Compustat--CCM panel; the frozen predictors log_me, book_to_market, and mom_12_2; the baseline VIX HIGH/LOW regime plus the LOW/MIDDLE/HIGH three-state extension; Elastic Net and XGBoost; strict expanding-window out-of-sample estimation; monthly rank diagnostics; and prediction-sorted economic-value analysis.

## How to Read This Appendix

Entries distinguish **General definition** from **Implementation in this research** whenever the distinction matters. Timing, source/implementation location, and status are included when needed. Statuses distinguish current production methods, historical preliminary specifications, and deferred roadmap terms.

Implementation-specific statements are governed by current code and frozen configuration. Older preliminary documents are not silently reinterpreted as current methodology.

## Acronym Index

| Acronym | Full term | Meaning in this project | Current status |
|---|---|---|---|
| AMEX | American Stock Exchange | Eligible U.S. listing venue. | Current data construction |
| AT | Assets -- Total | Compustat fallback input in stockholders' equity. | Current data construction |
| B/M | Book-to-Market | Positive BE divided by complete prior-December firm ME under the June convention. | Current baseline |
| BE | Book Equity | Compustat-based numerator of B/M. | Current baseline |
| CCM | CRSP/Compustat Merged | Effective-dated link history between GVKEY and PERMNO. | Current data construction |
| CEQ | Common/Ordinary Equity -- Total | Compustat fallback input in stockholders' equity. | Current data construction |
| CIZ | CRSP current integrated stock-data format | Current CRSP interface preferred by the source resolver; production uses crsp.msf_v2. | Current data construction |
| CRSP | Center for Research in Security Prices | Monthly security returns, prices, shares, identifiers, and market data. | Current data construction |
| CV | Cross-Validation | Forward-chaining model-selection validation in this repository. | Current baseline |
| D1 | Decile 1 | Lowest predicted-return portfolio in a formation month. | Current economic-value evaluation |
| D10 | Decile 10 | Highest predicted-return portfolio. | Current economic-value evaluation |
| D10-D1 | Decile 10 minus Decile 1 | Long D10 and short D1 spread. | Current economic-value evaluation |
| DT | Deferred Taxes and Investment Tax Credit | Component added in the BE formula. | Current data construction |
| EW | Equal Weighted | Equal target weights within a portfolio leg. | Current economic-value evaluation |
| GVKEY | Global Company Key | Compustat firm identifier. | Current data construction |
| HAC | Heteroskedasticity and Autocorrelation Consistent | Inference for monthly rank-IC and portfolio-return series. | Current inference |
| HIGH | High-volatility regime | VIX_t strictly above expanding historical median through t. | Current baseline |
| IC | Information Coefficient | Primarily monthly cross-sectional Spearman prediction/realization correlation. | Current evaluation |
| ITCB | Investment Tax Credit -- Balance Sheet | Fallback deferred-tax/ITC component. | Current data construction |
| LOW | Low-volatility regime | VIX_t less than or equal to expanding median through t. | Current baseline |
| LS | Long-Short | Long D10, short D1 portfolio position. | Current economic-value evaluation |
| LT | Liabilities -- Total | Compustat fallback input with AT. | Current data construction |
| MAE | Mean Absolute Error | Mean absolute forecast error. | Current evaluation |
| ME | Market Equity | Absolute price times shares outstanding. | Current baseline |
| ML | Machine Learning | Predictive modeling class used by the project. | Current baseline |
| MSE | Mean Squared Error | Mean squared forecast error. | Current evaluation |
| NASDAQ | Nasdaq Stock Market | Eligible U.S. listing venue. | Current data construction |
| NW | Newey--West | Bartlett-weight HAC estimator used for current monthly inference. | Current inference |
| NYSE | New York Stock Exchange | Eligible U.S. listing venue. | Current data construction |
| OOS | Out-of-Sample | Time-respecting prediction/evaluation after historical model selection. | Current baseline |
| PERMNO | Permanent Number | CRSP permanent security identifier. | Current data construction |
| PS | Preferred Stock | Component subtracted in BE. | Current data construction |
| PSTK | Preferred Stock -- Carrying Value | Preferred-stock fallback and CEQ-based SHE input. | Current data construction |
| PSTKL | Preferred Stock -- Liquidating Value | Second preferred-stock choice. | Current data construction |
| PSTKRV | Preferred Stock -- Redemption Value | First preferred-stock choice. | Current data construction |
| R² | Coefficient of Determination | Ordinary diagnostic and separately defined OOS benchmark comparison. | Current evaluation |
| RMSE | Root Mean Squared Error | Square root of MSE. | Current evaluation |
| SEQ | Stockholders' Equity -- Total | First-choice SHE field. | Current data construction |
| SHAP | SHapley Additive exPlanations | Model explanation method not used in current baseline results. | Deferred roadmap term |
| SHRCD | CRSP Share Code | Legacy SIZ common-share filter; current production CIZ path uses CIZ classifications. | Historical/legacy interface |
| SHE | Stockholders' Equity | Intermediate quantity in BE = SHE + DT - PS. | Current data construction |
| SSE | Sum of Squared Errors | Error sum used in OOS R². | Current evaluation |
| TXDB | Deferred Tax Balance | Fallback combined with ITCB when TXDITC is unavailable. | Current data construction |
| TXDITC | Deferred Taxes and Investment Tax Credit | First-choice deferred-tax/ITC field. | Current data construction |
| VIX | Cboe Volatility Index | Market-volatility variable defining formation-month regimes. | Current baseline |
| VW | Value Weighted | Portfolio weights proportional to positive finite me_lag. | Current economic-value evaluation |
| WRDS | Wharton Research Data Services | Access platform for licensed CRSP, Compustat, CCM, and preferred Cboe VIX data. | Current data construction |
| XGB / XGBoost | Extreme Gradient Boosting | Nonlinear boosted-tree baseline model. | Current baseline |

**Acronym collisions:** CV means cross-validation here, not coefficient of variation. IC means information coefficient in the rank-evaluation context. ME means market equity; BE means book equity.

## Core Research Concepts

### Asset pricing
**General definition:** Study of the relation between information, risks/characteristics, prices, and expected returns.

**Implementation in this research:** Empirical predictive asset pricing focused on whether observable stock characteristics rank next-month returns differently across volatility regimes.

### Stock-return predictability
**General definition:** Statistical dependence between information known before a return and the later realized return.

**Implementation in this research:** Each formation month t produces forecasts of calendar-month t+1 returns. Evidence is evaluated through forecast error, cross-sectional ranking, and portfolio sorts.

### Cross-sectional predictability
Predictive information that differentiates securities at the same formation date. The primary rank statistic is calculated within each month across stocks.

### Time-series predictability
Predictability of a series through time. It is not the primary empirical object of the production baseline, although regime labels and monthly evaluation statistics form time series.

### Predictor / characteristic / feature
A model input. The frozen baseline contains exactly log_me, book_to_market, and mom_12_2. "Characteristic" emphasizes economic interpretation; "feature" emphasizes model input.

### Target variable
The canonical target is next_month_return: the return in the next actual calendar month after formation.

### Forecast / prediction
Model-generated estimate of next_month_return. Production stores separate Elastic Net and XGBoost predictions.

### Formation month
Month-end t at which predictors and the contemporaneous VIX regime are attached and the forecast is formed. formation_date is canonical; it must agree with date in the validated baseline panel.

### Realization month
Calendar month t+1 in which the target is realized. realized_return_date must be exactly one calendar month-end after formation_date.

### Baseline specification
Frozen three-predictor production specification: log_me, book_to_market, mom_12_2, estimated with Elastic Net and XGBoost.

### Preliminary reduced-predictor specification
**Status:** Historical preliminary specification.

Earlier single-stock and reduced-cross-section work used log_me and mom_12_2 before Compustat/CCM B/M integration. Those results are not rewritten as if B/M had always been present.

### Complete baseline specification
**Status:** Implemented current production baseline.

The three-predictor specification including B/M. It is the basis for current headline empirical conclusions.

### Production specification
Full validated production panel, full eligible historical universe, production configuration, and current economic-value/inference pipeline.

### Quick-run configuration
Reduced computational validation using the same core feature definitions but a top-300 fixed formation-date universe, a shorter OOS window, and reduced tuning grids. Quick output is not a production empirical result.

### Expanding window
**General definition:** Training sample grows over time rather than discarding older eligible history.

**Implementation in this research:** For each OOS formation month, training uses all eligible historical rows with formation dates strictly earlier than the prediction month and target realization dates no later than that month.

### Rolling window
A fixed-length moving training window. It is not the frozen production estimator; when mentioned, it must not be confused with the current expanding design.

### Historical information set / historically observable
Information permitted at a date under project timing rules. It includes only features known by formation and historical outcomes already realized by that date.

## Data Sources and Identifiers

### WRDS -- Wharton Research Data Services
Access platform for licensed research databases. The project uses WRDS for CRSP and Compustat/CCM and prefers a WRDS Cboe source for VIX. Credentials remain outside source code and Git.

**Implementation:** src/rdsrp/data/wrds.py, accounting.py, vix.py.

### CRSP -- Center for Research in Security Prices
Monthly U.S. security returns, price, shares outstanding, market data, identifiers, and security/listing classifications.

**Production source:** crsp.msf_v2 in CIZ format.

### CIZ
CRSP's current integrated stock-data interface used by the production resolver. For msf_v2, the CIZ path uses security classifications rather than legacy SHRCD/EXCHCD filters.

### Compustat
Annual firm accounting fundamentals. Production uses comp.funda and, when fields are available, standard industrial, consolidated, domestic-population annual observations in USD.

### CCM -- CRSP/Compustat Merged
Effective-dated link history between Compustat firms and CRSP securities. Production uses crsp.ccmxpf_lnkhist, accepted link types LC/LU, and primary indicators P/C.

### Cboe
Source of VIX. The production build resolves Cboe data through WRDS (validated provenance cboe_all.cboe); code also supports the official Cboe historical VIX CSV as an authoritative fallback when the WRDS Cboe table cannot be resolved.

### PERMNO
CRSP permanent security identifier and authoritative longitudinal security key. It is preferred to ticker because tickers can change.

### GVKEY
Compustat firm identifier used in annual fundamentals and CCM linkage.

### Ticker
Human-readable trading symbol retained for readability. It is not the authoritative longitudinal join key.

### CCM link / effective dating
A GVKEY--PERMNO association must be effective on the accounting datadate: missing start/end values are treated as open on that side, otherwise linkdt <= datadate <= linkenddt.

### linktype
CCM link type. Current accepted values are LC and LU.

### linkprim
CCM primary-link indicator. Current accepted values are P and C; deterministic conflict resolution prefers P.

### linkdt / linkenddt
Beginning and ending dates of CCM link validity.

## Security Filters and Universe

### Common stock / ordinary common equity
Production CIZ extraction requires sharetype NS, securitytype EQTY, securitysubtype COM, U.S. incorporation, accepted corporate issuer type, regular-way condition, active trading status, and eligible primary exchange.

### SHRCD
Legacy CRSP share code. The SIZ fallback uses SHRCD 10/11, but current production CIZ eligibility is governed by CIZ security classifications.

### EXCHCD
Legacy CRSP exchange code used by the SIZ fallback: 1/2/3 for NYSE/AMEX/NASDAQ. Current CIZ production uses primaryexch N/A/Q.

### NYSE / AMEX / NASDAQ
Eligible U.S. primary listing venues.

### $5 price filter
Feature-month eligibility screen requiring absolute price at t to be at least $5. It is applied after return histories needed for target construction are assembled, so t+1 price cannot decide t eligibility.

### Stock universe
Set of securities eligible under source and security filters for a given research mode.

### Fixed universe
Quick/preliminary validation can select top-N securities at one configured formation date and follow that fixed set. This is not the production universe.

### Monthly eligible universe
Within the production history, observations must satisfy current security/price/predictor/target/regime requirements at the relevant date. Production top_n_by_market_cap is null.

## Return and Timing Variables

### RET / ret / return
CRSP monthly total-return field used for momentum and target construction. Values are decimal returns.

### RETX / retx
CRSP return excluding distributions, retained from source where available but not the baseline forecasting target.

### Current-month return
Return during formation month t. It can contribute to later characteristics but is not the target for a forecast formed at t.

### Next-month return / target return
next_month_return is constructed by a calendar-safe one-month lead of ret and represents return in t+1.

### Realized return
Observed target after formation. Production prediction artifacts call it actual_next_month_return.

### formation_date
Explicit month-end t for model inputs and prediction.

### realized_return_date
Month-end at which target is realized; must equal t+1.

### Delisting return
A distinct delisting-return adjustment is not separately constructed by the current feature code. The project uses the CRSP monthly return field supplied by the resolved source and does not claim a separate DLRET-combination rule.

## Stock Characteristics / Predictors

### Market Equity -- ME
**General definition:** Equity market capitalization.

**Implementation in this research:** ME = |Price| x Shares Outstanding. CRSP shrout is in thousands, so core monthly market_equity/me is in dollar-thousands.

**Implementation:** src/rdsrp/features/build.py.

### log_me
Natural logarithm of strictly positive market equity. It is the size predictor.

### me_lag
**General definition:** Lagged market equity.

**Implementation in this research:** Previous **actual calendar month's** reconstructed market equity, created after calendar-complete reindexing. A calendar gap yields a missing lag. This canonical feature overwrites any raw source field of the same name and is the validated VW portfolio weight.

### me_lag_date / value_weight_date
Explicit previous-month date associated with me_lag. Portfolio validation requires it to equal formation month minus one calendar month-end.

### me_dec
Complete prior-December **firm** market equity used in B/M. Security-level December market equity is converted from dollar-thousands to dollar-millions and summed across all required valid linked securities for the firm. An incomplete required denominator is left missing.

### Stockholders' equity -- SHE
Fallback hierarchy: SEQ; else CEQ + PSTK when available; else AT - LT.

### Preferred stock -- PS
Hierarchy: PSTKRV, then PSTKL, then PSTK. If all are missing, current baseline explicitly treats PS as zero.

### Deferred taxes and investment tax credit -- DT
TXDITC first; otherwise TXDB + ITCB with missing components treated as zero; if no such item exists, current baseline sets this component to zero.

### Book Equity -- BE
**General definition:** Accounting equity used in characteristic construction.

**Implementation in this research:** BE = SHE + DT - PS. Nonpositive BE is retained for lineage diagnostics but cannot produce a valid baseline B/M.

### Book-to-Market -- B/M
**General definition:** Book equity divided by market equity.

**Implementation in this research:** For characteristic year y, positive BE associated with accounting year y-1 is divided by complete firm-level December ME from y-1, after unit conversion to millions. The characteristic becomes active in June y.

B/M and market-to-book are conceptual inverses but are not interchangeable; this project uses B/M only.

### Accounting datadate
Compustat fiscal-period record date used for annual assignment and effective CCM-link checks.

### Fiscal year / fyear
Compustat fiscal-year identifier retained for lineage. Annual deduplication and project characteristic timing are keyed to the calendar year of datadate.

### fyr
Compustat fiscal-year-end month field retained for lineage.

### Accounting year
Calendar year of datadate after keeping the latest standard annual record per GVKEY/calendar year.

### Characteristic year
accounting_year + 1. It defines the June-to-May interval for which an annual accounting characteristic is active.

### accounting_available_from / June convention
June 30 of characteristic_year. An annual record cannot enter the monthly panel before this date. For an accounting datadate in year y-1, the characteristic is active June y through May y+1.

### Lagged accounting information
Annual fundamentals intentionally enter later than the reported fiscal-year-end date under the conservative June rule to prevent premature use.

### Momentum / MOM / mom_12_2
**General definition:** Past-return characteristic.

**Implementation in this research:** Product of (1 + return) from t-12 through t-2 minus 1, using eleven calendar-contiguous monthly observations. Months t-1 and t are excluded; no t+1 information enters.

## Volatility and Market Regimes

### VIX -- Cboe Volatility Index
Market-implied volatility index used as the regime variable. Daily data are converted to monthly frequency using the final observed VIX close in each calendar month.

### Expanding historical median
Median of monthly VIX observations from the start of available VIX history through and including t.

### HIGH regime
**Implementation in this research:** HIGH iff VIX_t > expanding median through t.

### LOW regime
Complement of HIGH: VIX_t <= expanding median through t.

### Regime timing
The regime observed at formation month t is attached to the forecast whose target is realized at t+1.

## Econometric and Statistical Concepts

### In-sample estimation
Model fitting on the permitted historical training observations.

### Out-of-sample -- OOS
**General definition:** Evaluation on observations not used to fit the tested model.

**Implementation in this research:** Prediction month t is outside its fitted historical sample; training formation dates are strictly earlier than t and training target realizations are no later than t. Hyperparameters are selected using pre-OOS history before January 2000.

### MSE -- Mean Squared Error
MSE = mean[(y - yhat)^2]. Used for model-selection validation and forecast evaluation.

### RMSE -- Root Mean Squared Error
RMSE = sqrt(MSE), in return units.

### MAE -- Mean Absolute Error
MAE = mean[|y - yhat|].

### Ordinary R²
Standard sklearn-style coefficient of determination retained as a diagnostic in a general evaluation utility. It is distinct from the headline OOS R².

### OOS R²
**General definition:** Relative squared-error improvement over a benchmark forecast.

**Implementation in this research:** R²_OOS = 1 - SSE_model / SSE_benchmark, with the pooled expanding historical-mean benchmark defined below.

A negative OOS R² does **not** mean negative accuracy. It means model squared forecast error exceeded the chosen historical benchmark over the evaluated observations.

### Correlation
Linear prediction/realization correlation is retained as a diagnostic; cross-sectional ranking primarily uses monthly Spearman correlation.

### Standard error
Estimated sampling uncertainty of a statistic.

### t-statistic
Estimate divided by its estimated standard error.

### Statistical significance
Assessment relative to a sampling distribution/p-value. Current manuscript language distinguishes within-regime evidence from direct LOW-minus-HIGH difference tests and does not infer a regime difference merely from significance in one regime.

### HAC / Newey--West
**Status:** Implemented current economic-value/inference stage.

Heteroskedasticity- and autocorrelation-consistent inference using Bartlett weights. The frozen baseline uses six monthly lags for portfolio-return and rank-IC time series and regime-difference regressions.

## Machine-Learning Concepts

### Elastic Net
**General definition:** Linear regression with combined L1 and L2 coefficient penalties.

**Implementation in this research:** Regularized linear benchmark inside a StandardScaler pipeline. Production selected alpha=0.01, l1_ratio=0.1, max_iter=20000.

### L1 / L2 regularization
L1 penalizes absolute coefficient size; L2 penalizes squared coefficient size. Their mixture is controlled by l1_ratio and overall strength by alpha.

### Coefficient shrinkage
Regularization pulling estimated coefficients toward zero to control model complexity.

### StandardScaler
Centers/scales predictor columns using statistics estimated on the training sample only. It is fitted inside the Elastic Net pipeline; full-sample standardization is not used.

### XGBoost -- Extreme Gradient Boosting
Sequential boosted decision-tree model that can represent nonlinearities/interactions. Production uses squared-error loss, fixed random seed, and selected tree/learning/sampling hyperparameters.

### Feature importance
Descriptive XGBoost estimator output saved by the baseline run. It is not interpreted as a structural causal effect.

### Elastic Net standardized coefficient
Descriptive coefficient from the scaled Elastic Net fit, saved at refits. It is not a causal coefficient.

### SHAP
**Status:** Deferred / not part of current production baseline.

Shapley-based model explanation method. It must not be described as a current result until implementation and validated outputs exist.

## Model Selection and Time-Respecting Validation

### Parameter
Quantity estimated by model fitting, such as an Elastic Net coefficient or tree leaf value.

### Hyperparameter
Externally chosen setting controlling regularization or model architecture, such as alpha, l1_ratio, max_depth, or n_estimators.

### Training set
Historically eligible observations used to fit a model.

### Validation set
Later historical month used to compare candidate hyperparameters during pre-OOS forward-chaining selection.

### Test / OOS set
Formation-month cross-section whose outcome is not used to fit the model that produces its prediction.

### Cross-validation / CV
Repeated validation across historical folds. Random K-fold CV is not used for the production time-series problem.

### Forward-chaining validation / time-series CV
Chronological validation where training months precede the validation month. The implementation also removes training rows whose target realization would not yet be observable at the validation date.

### Hyperparameter grid
Finite candidate settings from configs/production.yaml.

### Production tuning sample
Pre-OOS 1990--1999 history with a deterministic cap of 500 rows per historical month used only to make grid search tractable. Production monthly refits use the full eligible expanding history.

### Model refit
Re-estimation using currently eligible history. Frozen production refit cadence is monthly.

### Preprocessing pipeline
Ordered preprocessing plus estimator. For Elastic Net it contains StandardScaler then ElasticNet; scaling is training-only.

## Out-of-Sample Evaluation

### Pooled expanding historical mean benchmark
**Status:** Headline production benchmark.

At formation month t, every stock receives the pooled mean of all historical next-month stock returns whose realized_return_date is on or before t. The benchmark is recomputed from the eligible expanding training sample each month.

This differs from the stock-specific historical-mean benchmark used in an earlier AAPL validation.

### Benchmark forecast
Reference prediction used in OOS R². It must itself be historically observable.

### Naive benchmark
Generic label for a simple comparison forecast. In current headline results the specific benchmark is the pooled expanding historical mean; do not replace it with a zero-return or stock-specific benchmark without labeling a different specification.

### Leakage-safe training rule
For prediction month t: training formation_date < t and training realized_return_date <= t.

## Cross-Sectional Ranking

### Spearman rank correlation
Correlation of ranks; measures monotonic cross-sectional ordering without requiring a linear return relationship.

### Rank IC -- Information Coefficient
**Implementation in this research:** Within each formation month/model, Spearman correlation between predicted next-month return and realized t+1 return. Monthly ICs are then summarized through time and subjected to HAC inference.

### Pearson IC
Monthly cross-sectional linear correlation retained as a secondary diagnostic.

### Fraction positive
Fraction of months in which a monthly diagnostic or net portfolio return is greater than zero, depending on the table.

## Portfolio Construction

### Portfolio formation
Assignment of stocks to portfolios using information available at formation t and realization of return at t+1.

### Sort / single sort / predicted-return sort
One-dimensional sort on model-predicted next-month return.

### Decile
One of ten ordered prediction portfolios.

### D1
Lowest predicted-return portfolio.

### D10
Highest predicted-return portfolio.

### D10-D1 / long-short
Return of D10 minus return of D1. A positive realization is a positive spread for that month; it is not automatically evidence of statistical significance or implementable profitability.

### Tie-preserving decile assignment
Current production economic-value module uses average percentile ranks and maps those percentiles to deciles, keeping identical predictions in the same portfolio. Intermediate deciles may be empty under large ties, while both extreme legs are required.

The historical preliminary portfolio module used a different deterministic tie-handling rule and is documented as historical rather than silently rewritten.

### EW -- Equal Weighted
Each valid security in a portfolio leg receives equal target weight.

### VW -- Value Weighted
Target weights are proportional to positive finite me_lag. Invalid VW weights are excluded; the code does not fall back to EW.

### Monthly rebalancing
Portfolio targets are reconstructed each formation month from that month's predictions.

### Turnover
**Status:** Implemented.

For an existing leg, one-way turnover equals one half of the L1 distance between new target weights and prior weights after drifting them through realized security returns. A newly initiated leg has turnover one. Total long-short turnover is long-leg plus short-leg turnover.

### Transaction costs
**Status:** Implemented current robustness specification.

Default frozen specification charges 50 basis points one way per dollar of turnover. Net long-short return = gross long-short return - 0.005 x total turnover.

### Gross return
Portfolio return before transaction-cost deduction.

### Net return
Gross return after the implemented turnover-based transaction-cost deduction.

## Performance Metrics and Inference

### Mean monthly return
Arithmetic mean of monthly portfolio returns.

### Annualized arithmetic return
12 times mean monthly return. This is not geometric compounding.

### Annualized volatility
sqrt(12) times monthly sample standard deviation.

### Sharpe ratio
Current descriptive annualized Sharpe is sqrt(12) times mean monthly net return divided by monthly standard deviation, under the implementation's zero-risk-free-rate convention.

### Cumulative return
When older preliminary figures use cumulative arithmetic sums, they are not labeled wealth indices. Current glossary should not silently convert arithmetic cumulative plots into compounded wealth.

### HAC mean inference
Current economic-value code computes the mean, HAC standard error, t-statistic, and two-sided normal-approximation p-value with six lags by default.

### Regime-difference inference
LOW-minus-HIGH difference is estimated directly with a constant plus LOW indicator and HAC covariance. Direct differences are distinct from comparing two separate within-regime significance results.

## Data Validation and Leakage Controls

### Missing value
Unavailable or invalid field. Production models require complete finite predictor rows and do not perform model-stage imputation.

### Complete case
Observation with all required fields for a specified analysis.

### Baseline complete case
Frozen production eligibility requiring all three predictors, target, regime/VIX, timing fields, and feature-month price eligibility. The canonical panel stores baseline_complete_case and the downstream loader checks consistency.

### Security-month
One PERMNO at one monthly formation date.

### Firm-year
One firm/accounting-year observation after annual deduplication rules.

### Duplicate key
More than one row for a key expected to be unique. The canonical final panel requires zero duplicate PERMNO-date rows.

### One-to-one merge
Each key appears at most once on both sides.

### Many-to-one merge
Many monthly rows can map to one annual characteristic row. Production code uses merge validation where this is the intended cardinality.

### Many-to-many merge
Potentially multiple matches on both sides. CCM linkage temporarily permits this form before effective-date filtering and deterministic conflict resolution; it is not allowed to propagate as duplicate security-months.

### Merge validation
Programmatic assertion that expected join cardinality and uniqueness are preserved.

### Data lineage
Retention of identifiers, datadates, availability dates, source tables, characteristic years, and validation metadata so derived variables can be traced.

### Unit consistency
Explicit compatibility of measurement units. Monthly CRSP ME is dollar-thousands; me_dec is converted to dollar-millions before division into Compustat BE.

### Extreme observation
Unusually large/small value investigated for possible unit/data errors. Production extreme-value guards are diagnostics, not winsorization.

### Winsorization
**General definition:** Clipping values to selected distributional quantiles.

**Implementation in this research:** A helper exists, but the production baseline data construction does **not** winsorize or trim predictors. Diagnostic bounds are not transformations.

### Standardization
Rescaling to a standardized scale. Production raw panel is not full-sample standardized; Elastic Net StandardScaler is fitted only on its historical training sample.

### Normalization
Generic rescaling term. In this project it should not be used as a vague synonym for either unit conversion, StandardScaler, or portfolio-weight normalization; name the actual operation.

### Look-ahead bias
Use of information unavailable at the decision/prediction date. Safeguards include calendar-safe t/t+1 targets, June accounting assignment, effective-dated CCM links, prior-December ME, expanding VIX median, forward-chaining tuning, training-only scaling, and target-observability checks.

### Target leakage
Use of target information, directly or indirectly, in predictors/model fitting before it is observable. The OOS engine requires realized training targets to be observable by the current formation date.

### Data leakage
Broader unintended flow of validation/test/future information into training, preprocessing, tuning, or feature construction.

### Survivorship bias
Bias from retaining only securities that survive to a later date. Production source queries use contemporaneous historical security eligibility and do not select the full production sample from present-day survivors. The older quick fixed-universe design is a computational validation design, not the production universe.

### Selection bias
Distortion from sample-selection rules. Security classifications, the $5 feature-month price screen, complete-case requirements, and required accounting linkage define the empirical population and must be considered for external validity.

### Future information
Any observation whose project-defined availability is after the relevant formation/validation date.

### Information set
All data permitted to a model at a particular formation date.

## Software and Reproducibility Terms

### Random seed
Fixed pseudorandom initialization input. Production configuration uses seed 42; deterministic tuning sampling and XGBoost reproducibility depend on fixed seeds plus stable inputs/software.

### Deterministic execution
Execution intended to reproduce results from identical data/configuration, aided by fixed seeds, explicit sorting, frozen parameters, and timing rules.

### Cache
Local copy of expensive/licensed source data or derived outputs. Licensed row-level caches remain outside Git.

### Checkpoint
Restart artifact. Baseline monthly checkpoints are fingerprinted against production panel/configuration/selected parameters/implementation so incompatible checkpoints cannot be silently reused.

### Configuration
YAML settings controlling construction, models, OOS design, and evaluation.

### Quick configuration
Reduced-scale computational validation configuration; not the production empirical specification.

### Production configuration
configs/production.yaml and associated frozen contracts used for headline baseline estimation.

### Metadata / run metadata
Machine-readable record of sources, sample, selected hyperparameters, seed, benchmark, timing, and code state.

### Environment / virtual environment
Python execution context. Documented local environment is Python 3.11 in .venv on Windows.

### Package version
rdsrp release version in pyproject.toml; identifies software release, not data vintage by itself.

### Git commit hash
Content-addressed repository-state identifier. Production metadata records the commit when available.

## Future / Deferred Terms

### Fama--MacBeth regression
**Status:** Deferred roadmap term; not implemented in current production baseline.

Two-step cross-sectional regression framework. If introduced later, document the exact regression specification, timing, weighting, and inference convention.

### SHAP
**Status:** Deferred roadmap term; not implemented in current production baseline.

Shapley-based explanation method. It must not be described as a current result until implementation and validation artifacts exist.

### Alternative regime definitions
**Status:** Planned robustness direction.

The current headline definition is VIX relative to its expanding median. Alternative quantiles, thresholds, macro regimes, or related states are new robustness specifications.

### Richer characteristic sets
**Status:** Planned extension.

The frozen baseline contains exactly three predictors. Additional characteristics must be introduced as separate specifications rather than silently changing the baseline.

## Variable Reference

| Variable | Meaning | Source / construction | Timing | Used for |
|---|---|---|---|---|
| permno | CRSP permanent security identifier | CRSP | Persistent security key | Joins/uniqueness |
| ticker | Human-readable symbol | CRSP metadata | Formation metadata | Readability |
| gvkey | Compustat firm identifier | Compustat/CCM | Annual linkage | B/M lineage |
| date | Monthly panel date | CRSP month-end | t | Core key |
| formation_date | Explicit prediction date | Canonical validation | t | OOS timing |
| realized_return_date | Target realization date | Feature builder | t+1 | Leakage guard |
| ret | Current monthly total return | CRSP | t | Momentum/target source |
| retx | Return excluding distributions | CRSP | t | Source/diagnostic |
| next_month_return | Calendar t+1 return | Calendar-safe ret lead | t+1 | Baseline target |
| actual_next_month_return | Output name for realized target | OOS output | t+1 | Evaluation/portfolios |
| prc | Absolute monthly price | CRSP | t | ME/$5 screen |
| shrout | Shares outstanding, thousands | CRSP | t | ME |
| market_equity / me | abs(prc) x shrout | Feature builder | t | Size/diagnostics |
| log_me | Natural log positive ME | Feature builder | t | Predictor |
| me_lag | Previous-calendar-month ME | Calendar-safe shift | t-1 | VW weights |
| me_lag_date | Date of me_lag | Feature builder | t-1 | Weight audit |
| book_equity | SHE + DT - PS | Compustat | Annual/June-lagged | B/M numerator |
| accounting_datadate | Accounting record date | Compustat | Fiscal period | Lineage |
| accounting_available_from | June 30 characteristic year | Timing rule | First allowed date | Leakage guard |
| characteristic_year | accounting_year + 1 | Accounting prep | June--May | Monthly assignment |
| me_dec | Complete prior-December firm ME, millions | CRSP+CCM | Dec y-1 | B/M denominator |
| book_to_market | Positive BE / positive complete me_dec | Compustat+CRSP+CCM | Active June--May | Predictor |
| mom_12_2 | Compounded t-12 through t-2 return | CRSP history | Known by t | Predictor |
| vix | Last observed monthly VIX close | Cboe/WRDS | t | Regime |
| vix_expanding_median | Expanding median through t | Regime module | Through t | Threshold |
| regime | HIGH/LOW | VIX rule | t | Conditional evaluation |
| price_eligible | $5 feature-month flag | Validation | t | Sample screen |
| baseline_complete_case | Frozen model eligibility | Validation | t plus target/timing | Production sample |
| benchmark_prediction | Pooled mean of observable historical targets | OOS estimator | Recomputed at t | OOS R² |
| elastic_net_prediction | Elastic Net forecast | OOS estimator | Formed t | Evaluation |
| xgboost_prediction | XGBoost forecast | OOS estimator | Formed t | Evaluation |
| spearman_ic | Monthly cross-sectional Spearman IC | Reporting | Month/model | Rank inference |
| portfolio | Prediction decile 1--10 | Economic-value module | t | D1/D10 |
| gross_long_short_return | D10-D1 before costs | Economic-value module | t+1 | Economic performance |
| net_long_short_return | Gross spread less turnover cost | Economic-value module | t+1 | Cost-adjusted performance |

## Model Hyperparameter Reference

### Elastic Net

| Hyperparameter | Meaning | Current production role |
|---|---|---|
| alpha | Overall regularization strength | Frozen selected value 0.01 |
| l1_ratio | L1/L2 penalty mix | Frozen selected value 0.1 |
| max_iter | Optimization iteration cap | 20000 |
| random_state | Reproducibility seed | 42 |

Candidate values remain in configs/production.yaml. Selected values are also recorded in model-selection/run metadata.

### XGBoost

| Hyperparameter | Meaning | Current production role |
|---|---|---|
| n_estimators | Number of boosted trees | 200 |
| max_depth | Maximum tree depth | 2 |
| learning_rate / eta | Boosting shrinkage | 0.01 |
| subsample | Row sampling fraction | 1.0 |
| colsample_bytree | Column sampling fraction/tree | 0.7 |
| tree_method | Tree construction algorithm | hist |
| random_state | Reproducibility seed | 42 |
| n_jobs | Fitting threads | 4 |

## Mathematical Notation

- i = stock/security.
- t = formation month.
- t+1 = target-return month.
- R_i,t = stock i return in month t.
- X_i,t = frozen predictor vector at formation.
- yhat_i,t+1 = forecast of next-month return.
- ME_i,t = market equity.
- BE_i,y = annual book equity.
- ME^Dec_i,y = December market equity used for B/M.
- VIX_t = formation-month VIX.
- VIX-median_t = expanding VIX median through t.
- R^D10_t+1 and R^D1_t+1 = extreme-decile realized returns.
- R^LS_t+1 = D10-D1 long-short return.

The manuscript is authoritative if later stages introduce formal new notation; this appendix should be updated to match rather than inventing competing symbols.

## Canonical Terms and Historical Aliases

- formation_date is the canonical explicit prediction date; older paths sometimes use date for the same monthly key. In the current canonical panel they must agree.
- next_month_return is the canonical panel target; actual_next_month_return is the same economic realization after it enters OOS outputs.
- "Preliminary two-predictor specification" means log_me + mom_12_2. "Complete baseline" means all three frozen predictors including B/M.
- eta and learning_rate are equivalent XGBoost names in repository configuration/model APIs; current production documentation uses learning_rate.
- Newey--West and HAC refer to the current six-lag Bartlett-weight implementation unless a future method explicitly changes it.

## Three-State Volatility Regime Extension

### Binary Regime

The binary regime is the original baseline volatility classification. At formation month t, `regime_binary` is LOW when VIX is less than or equal to the expanding historical VIX median through t and HIGH otherwise. The historical `regime` variable remains the backward-compatible binary label and must not be changed by the extension.

### Three-State Volatility Regime

A tercile divides an ordered distribution into three parts using two percentile thresholds. The extended `regime_3state` classification uses expanding historical VIX terciles computed only from information available through and including the formation month. The project uses NumPy's deterministic `quantile(..., method="linear")` convention and the same current-month-included information set as the existing expanding median.

- `q33` / `vix_expanding_q33`: the 33.33rd-percentile expanding VIX threshold, $Q_{0.33,t}$.
- `q67` / `vix_expanding_q67`: the 66.67th-percentile expanding VIX threshold, $Q_{0.67,t}$.
- LOW: $VIX_t \leq Q_{0.33,t}$.
- MIDDLE: $Q_{0.33,t} < VIX_t \leq Q_{0.67,t}$.
- HIGH: $VIX_t > Q_{0.67,t}$.

MIDDLE therefore means an intermediate formation-time VIX state by threshold definition; it does not imply that model or portfolio performance must be numerically intermediate. The three-state regime is used only to condition evaluation of the already-frozen OOS predictions. It is not a model predictor, is not optimized using portfolio outcomes, and does not create separately trained regime-specific models.

## Implementation Locations

- CRSP/WRDS extraction: src/rdsrp/data/wrds.py
- Compustat/CCM extraction: src/rdsrp/data/accounting.py
- VIX extraction/monthly alignment: src/rdsrp/data/vix.py
- size/momentum/target timing: src/rdsrp/features/build.py
- book equity and B/M timing: src/rdsrp/features/book_equity.py
- VIX regime labeling: src/rdsrp/regimes/vix_regime.py
- canonical sample/benchmark contract: src/rdsrp/baseline/data.py
- tuning and expanding OOS estimation: src/rdsrp/baseline/estimation.py
- predictive metrics/rank IC: src/rdsrp/baseline/reporting.py
- portfolios/turnover/costs/HAC: src/rdsrp/baseline/economic_value.py
- three-state regime/economic-value extension: src/rdsrp/baseline/economic_value_three_regime.py
- production configuration: configs/production.yaml
- quick configuration: configs/quick.yaml
- manuscript methods: paper/sections/02_data.tex, 03_methods.tex, 05_economic_value.tex

## Change Log

| Research stage | Terminology added / updated |
|---|---|
| Initial market-data construction | WRDS, CRSP, CIZ, PERMNO, ME, VIX, return timing |
| Accounting integration | Compustat, CCM, GVKEY, BE, B/M, prior-December ME, June convention |
| Baseline modeling-panel validation | formation/realized dates, complete case, lineage, leakage controls |
| Full baseline estimation | Elastic Net, XGBoost, forward-chaining CV, expanding OOS, pooled benchmark, OOS R², Rank IC |
| Production economic-value evaluation | D1/D10/D10-D1, EW/VW, me_lag, turnover, 50-bps costs, HAC/Newey--West, Sharpe |
| Three-state volatility extension | expanding q33/q67, terciles, MIDDLE, regime_binary, regime_3state, deterministic prediction deciles |
| Future robustness/extension work | Update when new regimes, predictors, inference, or interpretability methods are implemented |

## Terminology Maintenance Requirement

Before every future research stage is considered complete, review this appendix for new or changed acronyms, variables, formulas, datasets, statistical methods, portfolio terms, timing rules, aliases, and implementation statuses. Run scripts/audit_terminology.py and reconcile README/paper wording when a canonical definition changes.

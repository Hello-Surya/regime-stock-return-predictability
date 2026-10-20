# Milestone 11 — Structured Literature Review Matrix

Search date: 2026-10-20. The accompanying CSV contains all required research fields. **Not established** is deliberate; abstracts do not support inventing sample dates, precise forecast horizons, or model details. Citation metadata and abstracts were checked against journal/publisher and academic-index pages. Most full texts were not examined.

## Priority reading and relationship to this project

### 1. GuKellyXiu2020 — Empirical Asset Pricing via Machine Learning
Shihao Gu, Bryan Kelly, Dacheng Xiu (2020), *The Review of Financial Studies*. [Verified record](https://doi.org/10.1093/rfs/hhaa009).

**Source-supported summary:** Compares machine-learning risk-premium forecasts; nonlinear interactions are important.

**Relation / novelty audit:** Closest: establishes ML/linear comparisons and economic gains. **Outcome:** Individual US stocks and aggregate; **regime design:** No explicit LOW/MIDDLE/HIGH VIX tercile test in verified description. This metadata-level review does not certify that all underlying samples, tests or implementation details are directly comparable.

### 2. AvramovChengMetzker2023 — Machine Learning vs. Economic Restrictions: Evidence from Stock Return Predictability
Doron Avramov, Si Cheng, Lior Metzker (2023), *Management Science*. [Verified record](https://doi.org/10.1287/mnsc.2022.4449).

**Source-supported summary:** Deep-learning profitability weakens on excluding microcaps and periods of high volatility and under realistic trading costs.

**Relation / novelty audit:** Critical overlap: volatility states; microcaps; costs. **Outcome:** Individual stocks; **regime design:** High-volatility exclusion studied. This metadata-level review does not certify that all underlying samples, tests or implementation details are directly comparable.

### 3. FreybergerNeuhierlWeber2020 — Dissecting Characteristics Nonparametrically
Joachim Freyberger, Andreas Neuhierl, Michael Weber (2020), *The Review of Financial Studies*. [Verified record](https://doi.org/10.1093/rfs/hhz123).

**Source-supported summary:** Nonparametric characteristic selection finds important nonlinearities and redundant candidate predictors.

**Relation / novelty audit:** Overlap: nonlinear characteristic structure. **Outcome:** Individual stocks; **regime design:** Not established from abstract. This metadata-level review does not certify that all underlying samples, tests or implementation details are directly comparable.

### 4. ChenPelgerZhu2024 — Deep Learning in Asset Pricing
Luyang Chen, Markus Pelger, Jason Zhu (2024), *Management Science*. [Verified record](https://doi.org/10.1287/mnsc.2023.4695).

**Source-supported summary:** Deep conditional asset-pricing approach extracts economic states and estimates cross-sectional return pricing.

**Relation / novelty audit:** Strong overlap on state-dependent flexible return modeling. **Outcome:** Individual stocks; **regime design:** Economy states learned from macro data; not identical to VIX terciles. This metadata-level review does not certify that all underlying samples, tests or implementation details are directly comparable.

### 5. KozakNagelSantosh2020 — Shrinking the cross-section
Serhiy Kozak, Stefan Nagel, Shrihari Santosh (2020), *Journal of Financial Economics*. [Verified record](https://doi.org/10.1016/j.jfineco.2019.06.008).

**Source-supported summary:** Economically motivated shrinkage of characteristics-based factors improves robust high-dimensional SDF estimation.

**Relation / novelty audit:** Method and baseline comparison rather than direct VIX competitor. **Outcome:** Individual stocks; **regime design:** Not established from abstract. This metadata-level review does not certify that all underlying samples, tests or implementation details are directly comparable.

### 6. KellyPruittSu2019 — Characteristics are covariances: A unified model of risk and return
Bryan T. Kelly, Seth Pruitt, Yinan Su (2019), *Journal of Financial Economics*. [Verified record](https://doi.org/10.1016/j.jfineco.2019.05.001).

**Source-supported summary:** Instrumented principal-component analysis models characteristics as time-varying risk exposures.

**Relation / novelty audit:** Overlap: time variation and characteristic modeling. **Outcome:** Individual stocks; **regime design:** Time-varying loadings, not discrete VIX regimes. This metadata-level review does not certify that all underlying samples, tests or implementation details are directly comparable.

### 7. MurrayXiaXiao2024 — Charting by machines
Scott Murray, Yusen Xia, Houping Xiao (2024), *Journal of Financial Economics*. [Verified record](https://doi.org/10.1016/j.jfineco.2024.103791).

**Source-supported summary:** ML forecasts from historical price patterns display cross-sectional power including large stocks.

**Relation / novelty audit:** Counterexample to presuming ML signals inherently fail in large stocks. **Outcome:** Individual stocks; **regime design:** Not established from abstract. This metadata-level review does not certify that all underlying samples, tests or implementation details are directly comparable.

### 8. Lewellen2015 — The Cross-section of Expected Stock Returns
Jonathan Lewellen (2015), *Critical Finance Review*. [Verified record](https://doi.org/10.1561/104.00000024).

**Source-supported summary:** Historical Fama-MacBeth combinations of stock characteristics produce informative next-return rankings.

**Relation / novelty audit:** Important simple benchmark for Milestone 12. **Outcome:** Individual stocks; **regime design:** Not a discrete volatility-regime study in verified abstract. This metadata-level review does not certify that all underlying samples, tests or implementation details are directly comparable.

### 9. GiglioKellyXiu2022 — Factor Models, Machine Learning, and Asset Pricing
Stefano Giglio, Bryan Kelly, Dacheng Xiu (2022), *Annual Review of Financial Economics*. [Verified record](https://doi.org/10.1146/annurev-financial-101521-104735).

**Source-supported summary:** Reviews ML asset pricing, factor models, evaluation, and inference.

**Relation / novelty audit:** Contextual survey. **Outcome:** Stock portfolios and individual stocks; **regime design:** Various; review rather than one forecast exercise. This metadata-level review does not certify that all underlying samples, tests or implementation details are directly comparable.

### 10. Bagnara2024 — Asset Pricing and Machine Learning: A critical review
Matteo Bagnara (2024), *Journal of Economic Surveys*. [Verified record](https://doi.org/10.1111/joes.12532).

**Source-supported summary:** Critical synthesis of ML asset pricing using regularization, trees, dimension reduction, and networks.

**Relation / novelty audit:** Contextual survey; not direct competitor. **Outcome:** Survey; **regime design:** Various; not one empirical sample. This metadata-level review does not certify that all underlying samples, tests or implementation details are directly comparable.

### 11. FamaFrench1992 — The Cross-Section of Expected Stock Returns
Eugene F. Fama, Kenneth R. French (1992), *The Journal of Finance*. [Verified record](https://doi.org/10.1111/j.1540-6261.1992.tb04398.x).

**Source-supported summary:** Documents strong associations of size and book-to-market with average cross-sectional returns.

**Relation / novelty audit:** Motivates frozen predictors. **Outcome:** Individual stocks; **regime design:** Not evaluated as VIX terciles. This metadata-level review does not certify that all underlying samples, tests or implementation details are directly comparable.

### 12. FamaFrench1993 — Common risk factors in the returns on stocks and bonds
Eugene F. Fama, Kenneth R. French (1993), *Journal of Financial Economics*. [Verified record](https://doi.org/10.1016/0304-405X(93)90023-5).

**Source-supported summary:** Studies common market, size and value factors.

**Relation / novelty audit:** Motivates risk adjustment; distinct from statistical forecasting. **Outcome:** Stocks and bonds; **regime design:** Not applicable. This metadata-level review does not certify that all underlying samples, tests or implementation details are directly comparable.

### 13. FamaFrench2015 — A five-factor asset pricing model
Eugene F. Fama, Kenneth R. French (2015), *Journal of Financial Economics*. [Verified record](https://doi.org/10.1016/j.jfineco.2014.10.010).

**Source-supported summary:** Adds profitability and investment factors to size and value for average-return explanation.

**Relation / novelty audit:** Shows omitted characteristic and factor risk considerations. **Outcome:** Individual stocks and factor portfolios; **regime design:** Not the same VIX regime exercise. This metadata-level review does not certify that all underlying samples, tests or implementation details are directly comparable.

### 14. JegadeeshTitman1993 — Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency
Narasimhan Jegadeesh, Sheridan Titman (1993), *The Journal of Finance*. [Verified record](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x).

**Source-supported summary:** Documents medium-horizon momentum profitability.

**Relation / novelty audit:** Motivates frozen momentum predictor. **Outcome:** Individual stocks; **regime design:** Not the same VIX regime exercise. This metadata-level review does not certify that all underlying samples, tests or implementation details are directly comparable.

### 15. NovyMarx2013 — The other side of value: The gross profitability premium
Robert Novy-Marx (2013), *Journal of Financial Economics*. [Verified record](https://doi.org/10.1016/j.jfineco.2013.01.003).

**Source-supported summary:** Gross profitability captures incremental cross-sectional expected-return information.

**Relation / novelty audit:** Additional predictor to consider in M13, not M11. **Outcome:** Individual stocks; **regime design:** Not established from abstract. This metadata-level review does not certify that all underlying samples, tests or implementation details are directly comparable.

### 16. HouXueZhang2020 — Replicating Anomalies
Kewei Hou, Chen Xue, Lu Zhang (2020), *The Review of Financial Studies*. [Verified record](https://doi.org/10.1093/rfs/hhy131).

**Source-supported summary:** Microcap mitigation, value weighting, and multiple testing substantially reduce anomaly significance.

**Relation / novelty audit:** High relevance to EW/VW divergence and multiple comparisons. **Outcome:** Individual-stock anomaly portfolios; **regime design:** Not a VIX state ML test. This metadata-level review does not certify that all underlying samples, tests or implementation details are directly comparable.

### 17. HarveyLiuZhu2016 — ... and the Cross-Section of Expected Returns
Campbell R. Harvey, Yan Liu, Heqing Zhu (2016), *The Review of Financial Studies*. [Verified record](https://doi.org/10.1093/rfs/hhv059).

**Source-supported summary:** Warns that many tests in asset pricing raise false-discovery risk.

**Relation / novelty audit:** Significance thresholds and multiplicity concerns. **Outcome:** Cross-sectional factor findings; **regime design:** Not applicable. This metadata-level review does not certify that all underlying samples, tests or implementation details are directly comparable.

### 18. McLeanPontiff2016 — Does Academic Research Destroy Stock Return Predictability?
R. David McLean, Jeffrey Pontiff (2016), *The Journal of Finance*. [Verified record](https://doi.org/10.1111/jofi.12365).

**Source-supported summary:** Anomaly returns attenuate out-of-sample and after publication.

**Relation / novelty audit:** Stability and genuine holdout motivation. **Outcome:** Individual stock portfolios; **regime design:** Not a VIX state ML test. This metadata-level review does not certify that all underlying samples, tests or implementation details are directly comparable.

### 19. DeMiguelEtAl2020 — A Transaction-Cost Perspective on the Multitude of Firm Characteristics
Victor DeMiguel, Alberto Martin-Utrera, Francisco J. Nogales, Raman Uppal (2020), *The Review of Financial Studies*. [Verified record](https://doi.org/10.1093/rfs/hhz085).

**Source-supported summary:** Transaction costs affect the number of economically useful stock characteristics.

**Relation / novelty audit:** Implementation quality and combined signal issues. **Outcome:** Individual-stock trading portfolios; **regime design:** Not established from abstract. This metadata-level review does not certify that all underlying samples, tests or implementation details are directly comparable.

### 20. NovyMarxVelikov2016 — A Taxonomy of Anomalies and Their Trading Costs
Robert Novy-Marx, Mihail Velikov (2016), *The Review of Financial Studies*. [Verified record](https://doi.org/10.1093/rfs/hhv063).

**Source-supported summary:** Trading costs and turnover can materially reduce anomaly returns.

**Relation / novelty audit:** Turnover-cost assumptions and economic interpretation. **Outcome:** Individual-stock anomaly portfolios; **regime design:** Not a VIX state ML test. This metadata-level review does not certify that all underlying samples, tests or implementation details are directly comparable.

### 21. BekaertHoerova2014 — The VIX, the variance premium and stock market volatility
Geert Bekaert, Marie Hoerova (2014), *Journal of Econometrics*. [Verified record](https://doi.org/10.1016/j.jeconom.2014.05.008).

**Source-supported summary:** Decomposes VIX into variance and variance-premium components; studies aggregate-return predictability.

**Relation / novelty audit:** VIX is multi-mechanism indicator; not causal treatment. **Outcome:** Aggregate stock market; **regime design:** VIX-based decomposition; no verified stock-level terciles. This metadata-level review does not certify that all underlying samples, tests or implementation details are directly comparable.

### 22. BollerslevTauchenZhou2009 — Expected Stock Returns and Variance Risk Premia
Tim Bollerslev, George Tauchen, Hao Zhou (2009), *The Review of Financial Studies*. [Verified record](https://doi.org/10.1093/rfs/hhp008).

**Source-supported summary:** Variance-risk premia help predict aggregate market returns, notably at intermediate horizons.

**Relation / novelty audit:** Conceptual VIX/return link; target differs. **Outcome:** Aggregate market return; **regime design:** Implied versus realized variance; not tercile ranking. This metadata-level review does not certify that all underlying samples, tests or implementation details are directly comparable.

### 23. MoreiraMuir2017 — Volatility-Managed Portfolios
Alan Moreira, Tyler Muir (2017), *The Journal of Finance*. [Verified record](https://doi.org/10.1111/jofi.12513).

**Source-supported summary:** Volatility-scaled factor exposures can change Sharpe ratios.

**Relation / novelty audit:** VIX states affect risk and implementability, but task differs. **Outcome:** Factor portfolios; **regime design:** Continuous volatility management. This metadata-level review does not certify that all underlying samples, tests or implementation details are directly comparable.

### 24. DanielMoskowitz2016 — Momentum crashes
Kent Daniel, Tobias J. Moskowitz (2016), *Journal of Financial Economics*. [Verified record](https://doi.org/10.1016/j.jfineco.2015.12.002).

**Source-supported summary:** Momentum crashes cluster in panic states following market declines with high volatility.

**Relation / novelty audit:** Strong potential mechanism / confounding for momentum input. **Outcome:** Momentum portfolios; **regime design:** Panic states and high volatility. This metadata-level review does not certify that all underlying samples, tests or implementation details are directly comparable.

### 25. BarrosoSantaClara2015 — Momentum has its moments
Pedro Barroso, Pedro Santa-Clara (2015), *Journal of Financial Economics*. [Verified record](https://doi.org/10.1016/j.jfineco.2014.11.010).

**Source-supported summary:** Momentum risk varies through time and can be managed.

**Relation / novelty audit:** Conditional momentum benchmarks. **Outcome:** Momentum portfolios; **regime design:** Time-varying volatility risk. This metadata-level review does not certify that all underlying samples, tests or implementation details are directly comparable.

### 26. CampbellHentschel1992 — No news is good news: An asymmetric model of changing volatility in stock returns
John Y. Campbell, Ludger Hentschel (1992), *Journal of Financial Economics*. [Verified record](https://doi.org/10.1016/0304-405X(92)90037-X).

**Source-supported summary:** Models volatility feedback and asymmetric aggregate market returns.

**Relation / novelty audit:** Mechanism background; different outcome. **Outcome:** Aggregate market return; **regime design:** Volatility level continuously modeled. This metadata-level review does not certify that all underlying samples, tests or implementation details are directly comparable.

## Other verified records

- Ivo Welch, Amit Goyal (2008). [A Comprehensive Look at The Empirical Performance of Equity Premium Prediction](https://doi.org/10.1093/rfs/hhm014) — Documents fragile aggregate equity-premium out-of-sample prediction.
- John Y. Campbell, Samuel B. Thompson (2008). [Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average?](https://doi.org/10.1093/rfs/hhm055) — Small aggregate OOS predictability can have economic value under restrictions.
- David E. Rapach, Jack K. Strauss, Guofu Zhou (2010). [Out-of-Sample Equity Premium Prediction: Combination Forecasts and Links to the Real Economy](https://doi.org/10.1093/rfs/hhp063) — Forecast averaging mitigates uncertainty in aggregate premium prediction.
- Francis X. Diebold, Roberto S. Mariano (1995). [Comparing Predictive Accuracy](https://doi.org/10.1080/07350015.1995.10524599) — Establishes loss-differential predictive-accuracy comparisons under dependence assumptions.
- Raffaella Giacomini, Halbert White (2006). [Tests of Conditional Predictive Ability](https://doi.org/10.1111/j.1468-0262.2006.00718.x) — Conditional predictive ability testing allows forecast estimation and state conditioning under specific asymptotics.
- Whitney K. Newey, Kenneth D. West (1987). [A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix](https://doi.org/10.2307/1913610) — HAC covariance estimation for dependent residuals.
- Sture Holm (1979). [A Simple Sequentially Rejective Multiple Test Procedure](https://www.jstor.org/stable/4615733) — Step-down family-wise error control for multiple hypotheses.
- Yoav Benjamini, Yosef Hochberg (1995). [Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) — Controls the expected proportion of false rejections among discoveries under relevant assumptions.
- Daniele Bianchi, Matthias Buchner, Andrea Tamoni (2021). [Bond Risk Premiums with Machine Learning](https://doi.org/10.1093/rfs/hhaa062) — Examines ML prediction of bond returns; published corrigendum addressed future-information availability.
- Daniele Bianchi, Matthias Buchner, Tobias Hoogteijling, Andrea Tamoni (2021). [Corrigendum: Bond Risk Premiums with Machine Learning](https://doi.org/10.1093/rfs/hhaa098) — Documents the effect of correcting forward-information use in prior bond ML forecasting.
- Xinyue Fang, Robert Slepaczuk (2026). [Volatility Forecasting and Return Prediction under Market Regimes: Evidence from High-Frequency Chinese Equity Data](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6898918) — Reports a regime-augmented volatility/return ML approach with strictly walk-forward XGBoost in a Chinese index context.

## Reading priorities

1. Obtain and examine the full papers of Avramov--Cheng--Metzker, Gu--Kelly--Xiu, Chen--Pelger--Zhu, Hou--Xue--Zhang, Freyberger--Neuhierl--Weber, and the volatility/momentum studies before claiming uniqueness.
2. Extract samples, estimation protocols, exact volatility measures, costs, and precise model comparison tests from full text, updating unknown CSV fields rather than guessing.
3. Keep *aggregate* market risk-premium papers distinct from *cross-sectional* individual-stock forecasting work.
4. Screen for later versions and papers through the eventual submission date; the present search cannot establish absolute priority or exhaustive originality.

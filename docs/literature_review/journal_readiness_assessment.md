# Journal Readiness and Referee-Style Evaluation
Date: 2026-10-20. Judgment is contingent on full-text literature audit and future tests. A software artifact is not the same as a significant, new or journal-publishable economic finding.

## Provisional position: proceed but narrow the contribution
A focused conditional forecasting and economic-implementation study may be defensible if Milestones 12–17 show value beyond the results of Gu–Kelly–Xiu (2020), Freyberger–Neuhierl–Weber (2020), and especially Avramov–Cheng–Metzker (2023). The existing 2000–2025 sample shows positive within-LOW equal-weighted XGBoost net spread returns, but the Holm-adjusted state contrasts are inconclusive and the result vanishes under value weighting. The paper should **not** be described as already demonstrating significant volatility-state heterogeneity or investable profitability.

## Scorecard: evidence now versus requirements
| Dimension | Current state | Missing evidence and milestones |
|---|---|---|
| Question importance | Well-motivated conditional cross-sectional predictability | Explain what is learned over Avramov et al. (M11 full-text/M18) |
| Originality | Substantial prior-work overlap | Distinct precise estimand, compare exact competing designs (M11/M18) |
| Data credibility | Documented 1990–2025 CRSP CIZ–Compustat–CCM timing and filters | Independent WRDS pull/reproduction; delisting/availability audit (M18) |
| Forecast integrity | Fixed 1990–99 hyperparameter selection; expanding OOS | Strong historical-mean/characteristic benchmarks, fair paired losses (M12) |
| Econometric inference | M9 HAC with Holm; joint tests may be added | Dependence, asymptotics, blocks, family-level multiplicity (M12/M16) |
| Regime definition | Binary expanding median and ternary expanding terciles | Alternative thresholds, realized volatility, VIX levels (M14) |
| Investability | M10 50-bp model plus turnover accounting | Microcaps, NYSE breakpoints, liquidity, borrow, nonlinear impact, capacity (M15) |
| External validity | 311 OOS months covering many market states | Subperiods, rolling windows, holdouts, regime-transition sensitivity (M16) |
| Economic interpretation | Observational VIX associations | Mechanism proxies, factor exposures, interpretability (M17) |
| Reproducibility | Open code and documentation; row-level WRDS restricted | Complete protected-input hash/refresh audit, locked environment, clean build (M18) |

## Plausible outlets and fit (not acceptance promises)
These are candidate editorial domains, **not** claims of current journal acceptance requirements. Authors must verify each journal's current aims/scope/instructions before submission.

| Outlet | Subject-matter fit | Likely scientific standard / concern | What would strengthen submission |
|---|---|---|---|
| *Journal of Empirical Finance* | Empirical cross-sectional stock-predictability and costs | Incremental insight beyond known ML results | Strong direct tests, trading implementation, competing mechanisms |
| *Journal of Banking & Finance* | Empirical finance, forecasting and investment consequences | Generalizability, economic relevance and robustness | Larger set of credible benchmark models, factor controls and robustness |
| *Quantitative Finance* | Financial forecasting methods, economic strategy evaluation | Significance of modest OOS metrics; replicability | Robust data design, sensitivity, benchmarks, carefully defined costs |
| *Journal of Financial Markets* | Microstructure/market efficiency and trading frictions | Pure monthly forecasting may be less direct fit; implementation mechanism underdeveloped | Explicit friction and market structure/capacity evidence |
| *Journal of Financial and Quantitative Analysis* | Quantitative asset pricing and econometrics | High bar for economically meaningful differentiation | A materially new result beyond Avramov and much stronger inference |
| *Review of Financial Studies*, *Journal of Financial Economics*, *Journal of Finance* | Broad frontier asset pricing | Very strong originality and interpretation expectations | Clear novel causal/theoretical or broadly influential empirical insight absent at present |

Publication venue selection is **not** yet justified by acceptance rates or editorial claims. Conduct a current author-guide and recent-articles check before final journal selection.

## Referee objections and proposed answer/evidence
| Referee question | Does the current work answer it? | Remediation |
|---|---|---|
| Why is this new? | No; direct competitor has considerable overlap | Full-text tabulation of Avramov and newer studies, narrow contribution (M11/M18) |
| What beyond standard ML asset pricing? | Partly: a uniform three-state monthly conditional-evaluation protocol | Paired simple-benchmark/within-month IC and loss differences (M12) |
| Why VIX rather than alternatives? | Historically available and observable, not yet uniquely justified | VIX vs realized volatility/variance-risk-premium proxies (M14) |
| Why three states and where are cutoffs from? | Implemented expanding terciles; no proof of optimality | Threshold/winsorization/rolling quantiles and transition sensitivity (M14) |
| Is there direct statistically significant regime heterogeneity? | No: Holm-adjusted three-state net pairwise tests do not reject | Joint tests, block-bootstrap CIs, transparent negative results (M12/M16) |
| Does XGBoost overfit? | Expanding OOS mitigates but does not rule out | Simpler and richer benchmarks, sample splits, ablations (M12/M13/M16) |
| Are gross/net returns driven by microcaps? | EW strong and VW weak suggests an issue, but is not proof | Size/microcap/liquidity screens, NYSE breakpoints, short feasibility (M15) |
| Are the costs realistic? | 50-bp proportional assumption, not observed execution | Borrow, financing, spread/impact/capacity scenarios (M15) |
| Is sample unusual? | 2000–2025 OOS is substantial but not invariant | Subperiods and independent holdout where feasible (M16) |
| Were hypotheses chosen after viewing output? | Some are retrospective and data-informed | Dated transparency protocol + all tested outcomes (M11) |
| Why does volatility affect forecasting? | No causal evidence | Carefully tested competing mechanisms with available proxies (M17) |
| Is a negative/null result worth publishing? | Potentially if precision and comparative design meaningfully resolve a dispute | Statistical power and benchmark-conclusive null intervals (M12–M18) |

## Current scientific decision
**Proceed but narrow the contribution.** Present this as a measurement-and-robustness assessment of conditional statistical and economic predictability, rather than a discovery of a new market-state effect. If stronger benchmarks and microcap screens eliminate the spread, consider an informative null / limits-to-implementation result; a null result alone is not automatically publishable. Do not pursue Milestone 12 before the M11 literature/reference verification and acceptance items are explicitly reviewed.

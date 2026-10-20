# Formal Research Questions, Estimands and Statistical Hypotheses

**Scientific status:** This protocol is written **after** inspecting M1–M10 production results (2026-10-20). It is a prospective plan for *additional* analyses, **not** preregistration of the 2000–2025 OOS record. Earlier hypotheses are documented retrospectively where supported; exploratory directions are clearly labeled.

## Central question
Does the statistical and economic value of cross-sectional next-month US stock-return forecasts differ by historically observed VIX volatility regime, and do the **relative** merits of nonlinear XGBoost versus regularized linear Elastic Net differ across those regimes?

**Secondary:** (a) Do regime-conditional predicted-return rankings differ after direct simultaneous comparisons? (b) Do conditional signal rankings translate into net EW and VW D10–D1 portfolio performance? (c) Are any differences robust to alternative regimes, simple benchmarks and investability filters?

**Exploratory:** Intermediate-volatility peak; causal/economic explanations; capacity and predictability heterogeneity by subgroup.

## Common estimand definitions and conventions
At formation month t, let \(S_t\in\{L,M,H\}\) be its historically available expanding-VIX-tercile state. Forecasts \(\widehat R^m_{i,t+1}\) for models \(m\in\{EN,XGB\}\) are fixed outputs of the baseline. Let \(IC^m_t=\operatorname{Spearman}_i(\widehat R^m_{i,t+1},R_{i,t+1})\), after fixed stock eligibility screens. Let \(G^m_{w,t}=R^{m,D10}_{w,t+1}-R^{m,D1}_{w,t+1}\), with weighting \(w\in\{EW,VW\}\). Let \(C^m_{w,t}(50)\) be the documented proportional trading-cost deduction **for both legs**. \(N^m_{w,t}=G^m_{w,t}-C^m_{w,t}(50)\). Let \(L^m_t=|I_t|^{-1}\sum_{i\in I_t}(R_{i,t+1}-\hat R^m_{i,t+1})^2\), an equal-stock-weight *monthly* squared loss, only for months in which both models and truth exist. Equal-date averaging is the declared comparison unit to respect cross-sectional dependence.

For any monthly outcome \(Y_t\), \(\mu^Y_r=E[Y_t\mid S_t=r]\) denotes the *conditional time-series mean*. This expectation is associational, not a causal potential outcome. Mean rank IC, mean gross/net returns, and model-loss differences are separate estimands; never use one as a proxy for another.

**Direct regime test:** regress a single monthly time series on intercept + indicators for M/H (L reference); jointly test \((\beta_M,\beta_H)=(0,0)\) with covariance-consistent Wald inference and consider the three explicit pairwise contrasts. Use an HAC covariance that accounts for serial dependence, with the existing six-month lag as a documented baseline sensitivity; extend with block/bootstrap and alternate bandwidth in M16. Pairwise p-values within a prespecified outcome/model/weighting family use Holm family-wise adjustment; across multiple families transparently inventory tests and report unadjusted and adjusted results. For month-level model comparisons, align the same dates and evaluation cross-section.

## H1 — Regime heterogeneity of cross-sectional rankings
- **Motivation and literature:** Expected-return relations can be state-dependent (Gu–Kelly–Xiu 2020; Chen–Pelger–Zhu 2024), but evidence need not imply VIX-tercile differences.
- **Null:** \(H_{0,1}: \mu^{IC^m}_L=\mu^{IC^m}_M=\mu^{IC^m}_H\), separately by model. **Alternative:** at least one differs (non-directional).
- **Observable metric:** Monthly Spearman IC; units correlation per formation month. **Data:** protected monthly predictions and realized stock returns, historical VIX labels, fixed stock screens.
- **Test:** joint two-dummy regression Wald/HAC; three signed pairwise differences with Holm within model; sensitivity to dependence and regime clustering.
- **Reject:** adjusted family-wise 5% rejection of joint/pairwise difference for the **declared** family; negative or statistically insignificant within one state is not proof that regimes differ.
- **Inconclusive:** wide HAC intervals, bandwidth sensitivity or no adjusted differences.
- **Milestones:** M9 existing evidence; M16 sensitivity; M18 report. **Status:** *Retrospective* for already-computed baseline contrasts; only new independence-sensitive checks can be pre-specified now.

## H2 — Relative XGBoost versus Elastic Net advantage varies by state
- **Motivation:** Nonlinear characteristic interactions (Gu–Kelly–Xiu 2020; Freyberger–Neuhierl–Weber 2020).
- **Primary relative-ranking estimand:** \(\Delta^{IC}_r=E[IC^{XGB}_t-IC^{EN}_t\mid S_t=r]\). **Null:** all three \(\Delta^{IC}_r\) equal; **alternative:** at least one difference.
- **Secondary accuracy estimand:** \(\Delta^L_r=E[L^{EN}_t-L^{XGB}_t\mid S_t=r]\); positive favors XGBoost (reduced loss). **Null:** all three values equal. Separately assess \(H_0:\Delta^L_r=0\) only when stated in the testing register.
- **Data:** matched calendar-month same-stock forecast/realization panel, baseline trained predictions and S_t; do **not** retrain during M11.
- **Test:** paired monthly model-difference series regressed on regime indicators; HAC Wald and three Holm pairwise contrasts. Investigate Diebold–Mariano and Giacomini–White applicability in M12 rather than mechanically treating all pooled forecast losses as independent.
- **Dependence:** monthly aggregation avoids the invalid fiction that thousands of stocks form independent OOS time observations; common shocks, estimation overlap and VIX persistence can still matter. Report cluster/block sensitivity.
- **Reject/inconclusive:** adjusted direct differences / relevant joint test versus no difference; near-zero conditional advantages or wide confidence intervals are inconclusive.
- **Milestones:** M12 new benchmark and paired model comparisons, M16 robust inference.
- **Status:** *Forward planned after historical data inspection*; cannot retroactively call confirmatory out-of-sample evidence.

## H3 — Conditional net economic value
- **Motivation:** Economic gains do not follow mechanically from model fit (Novy-Marx–Velikov 2016; Avramov–Cheng–Metzker 2023).
- **Primary declared net outcome:** XGBoost equal-weighted 50-bp net monthly D10–D1 in LOW VIX tercile. \(H_{0,3}:\mu^{N^{XGB}_{EW}}_L\le0\) versus \(H_{A,3}:\mu^{N^{XGB}_{EW}}_L>0\) if a one-sided test is used prospectively; **existing published repository p-values are two-sided** and must not be silently converted.
- **Secondary outcomes:** net VW returns; gross EW/VW; HIGH−LOW and HIGH−MIDDLE contrasts; risk-adjusted factor alpha and different cost rates. Fixed 50-bp scenario is **not** a true execution-cost estimate.
- **Test:** HAC mean and CIs; joint and Holm-adjusted pairwise net-return differences as separate families; portfolio factor regressions/funding and capacity checks later.
- **Reject/inconclusive:** only appropriately adjusted evidence of positive return supports a *statistical mean* claim; implementation requires borrow, impact, capital and capacity assessment. Nonpositive VW results block strong investability assertions.
- **Milestones:** M10 existing two-sided results; M15 new implementability; M17 factor-risk adjustment.
- **Status:** *Retrospective* for observed EW LOW performance, not a newly independent hypothesis.

## H4 — Intermediate-volatility peak (exploratory)
- **Motivation:** NON-MONOTONIC pattern was inspected in M8–M10, so it cannot be called prespecified.
- **Contrasts:** \(d_1=\mu^Y_M-\mu^Y_L\), \(d_2=\mu^Y_M-\mu^Y_H\). For a strict peak, \(H_A:d_1>0\ \mathrm{and}\ d_2>0\) for one declared metric, versus composite null at least one contrast nonpositive.
- **Metric:** primary exploratory \(Y=N^{XGB}_{EW}\), replicated for IC only if distinctly labeled and multiplicity reported.
- **Test:** pre-labeled one-sided intersection-union approach or appropriate joint simultaneous bounds with HAC covariance; do not infer the peak from two significant isolated means. Account for state persistence.
- **Interpretation:** Nonlinearity in performance does not imply nonlinear structural VIX causality.
- **Milestones:** M14 threshold robustness; M16 genuinely untouched periods if feasible.
- **Status:** *Exploratory/post hoc*; no valid claim of prospective confirmation from the already-seen sample.

## H5 — Investability survival
- **Motivation:** Small hard-to-arbitrage shares often explain apparent ML profitability (Avramov–Cheng–Metzker 2023; Hou–Xue–Zhang 2020).
- **Outcome:** net LOW-tercile D10–D1 on predeclared investable screens, preferably NYSE breakpoints / minimum market equity / dollar trading volume / size stratification with explicit long/short borrow treatment.
- **Null:** \(\mu_{\mathrm{net},\mathrm{investable},L}\le0\) versus \(>0\), tested separately for EW and VW, with exact cost/accounting assumptions pre-set.
- **Methods:** HAC inference, matched-position turnover costs, factor adjustment and block/subperiod sensitivity; report number of traded names, size share, cost assumptions, and full screen definition. Reject only with adjusted evidence and defensible exposure feasibility.
- **Milestones:** M15, M16, M18. **Status:** *Prospectively specified analysis plan after observing original sample*, not validated on independent data.

## Exploratory mechanisms / interpretation
Hypotheses of signal-to-noise, attention, funding liquidity, changing factor exposure and nonlinear interactions are **associations** to be distinguished using M17 proxies/interactions/ablation. VIX states are observational and are never treatments.

## Register and decision discipline
| Family | Outcome | Models/weights | Primary error control | Evidence stage |
|---|---|---|---|---|
| H1 | IC, direct 3-state differences | Each model | Holm on three pairwise p-values within each model, plus joint Wald | Existing M9 (retrospective) |
| H2 | XGB−EN paired monthly IC and loss | Matched models | Prespecified families and Holm pairwise + joint Wald | New M12 after preview of historical results |
| H3 | Net and gross spread returns | EW/VW per model | Report family-wise Holm within outcome/model/weight; inventory broader multiplicity | Existing M10 + M15 |
| H4 | MIDDLE greater than LOW and HIGH | Declared XGB EW net outcome | Intersection-union / simultaneous one-sided HAC intervals | Exploratory |
| H5 | Net returns after investability screens | Each predeclared screen + model/weight | Predeclared adjustment across screen families | New M15 |

**No p-value is proof of causality, profitable execution, or population invariance.** Maintain a dated decision log before new analyses. Negative and unstable results must remain in all tables.

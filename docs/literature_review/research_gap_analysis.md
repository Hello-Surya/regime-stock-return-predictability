# Research Gap and Originality Assessment

## Already established
- Flexible/nonlinear ML can improve cross-sectional expected-return modeling (Gu–Kelly–Xiu 2020; Freyberger–Neuhierl–Weber 2020; Chen–Pelger–Zhu 2024).
- Volatility and risk conditions vary over time and matter for return/risk processes (Bekaert–Hoerova 2014; Moreira–Muir 2017).
- ML investment returns can rely on difficult-to-arbitrage stocks and deteriorate under high volatility, microcap restrictions and costs (Avramov–Cheng–Metzker 2023; Jensen–Kelly–Malamud–Pedersen 2026).
- Economically meaningful anomaly claims must survive trading frictions, value weighting, replication and multiple testing (Novy-Marx–Velikov 2016; Hou–Xue–Zhang 2020; Harvey–Liu–Zhu 2016).
- Different return targets matter: forecasting the aggregate market equity premium is not equivalent to predicting rankings of **individual stocks**.

## Possible substantive question not resolved by the present audit
Under the **same three-characteristic information set**, do historically observable LOW/MIDDLE/HIGH VIX states reliably modify monthly *cross-sectional ranking*, model-to-model forecast advantage, and *net* long–short returns, compared with simple historical-mean and characteristic-based predictors? And is any state contrast robust to alternative volatility measures, NYSE-breakpoint portfolios, liquidity constraints, risk adjustment and time subperiods?

This is a **candidate** gap, not an established exclusive discovery; compare detailed methods of Avramov–Cheng–Metzker and other new articles before promoting it to the abstract.

## What current evidence supports
- Positive within-LOW tercile EW net XGBoost results (0.6135% per month, p=0.0143 at stylized 50-bp one-way trading costs).
- Three-state EW descriptive strength in MIDDLE, with MIDDLE p=0.0714 for net returns, and negative HIGH mean.
- No Holm-adjusted pairwise regime net return evidence at 5%; none of the three-state VW net means are positive for XGBoost.
- Almost zero overall return-level OOS R².
- These numbers concern **fixed forecasts evaluated by state**; no separate regime-specific estimator is fitted.

## Threats
1. **Priority:** existing state-dependent ML finance research may preempt a broad novelty statement; Avramov et al. especially; Jensen et al. (2026) additionally constrain novelty around cost-aware portfolios.
2. **Multiplicity:** favorable states, weightings and models may have been inspected before the present formal hypothesis protocol.
3. **Statistical power:** 104/108/99 state-specific months and temporally clustered uncertainty.
4. **EW–VW divergence:** dependence on potentially illiquid or microcap names and unmodeled shorting frictions.
5. **Benchmark omission:** absent stronger historical mean, OLS/characteristic and factor-based benchmark tests.
6. **VIX endogeneity:** regime labels are correlated with many macroeconomic conditions, not causal interventions.
7. **Threshold discretion:** terciles are one choice among many; the middle-state pattern was viewed before this documentation.
8. **External validity:** 2000–2025 US stock sample may reflect post-2000 structure, crisis windows and time evolution.

## What would substantiate differentiation
M12: strong simple forecast benchmarks and paired conditional loss/IC contrasts; M13: predictor expansions/ablation; M14: alternative VIX windows/measures and regime boundaries; M15: microcap, size and liquidity-capacity constraints, borrow/cost sensitivity; M16: genuinely separated subperiods and dependence-robust inference; M17: characteristics, factor exposures and competing economic mechanisms; M18: updated competitor review, reproduction and referee audit.

## Language controls
**Do not claim:** first demonstration; direct regime differences proven significant; causal volatility effect; implementable arbitrage; robust outperformance of linear benchmarks; an ex ante confirmatory intermediate-volatility maximum. Preferred: “document a descriptive conditional pattern and evaluate whether it is stable and economically meaningful after stronger controls”.

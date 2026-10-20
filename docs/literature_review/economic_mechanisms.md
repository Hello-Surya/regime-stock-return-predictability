# Economic Mechanisms and Falsifiable Implications

Every mechanism below is a **competing associational explanation** of conditional forecast quality, not a causal conclusion. VIX is a simultaneously determined market index, and categorizing its realization at t does not randomize market stress. Proxy availability and publication-time dating must be audited before M17.

| Mechanism | Supporting/adjacent verified literature | Potential observable prediction | Competing explanations / required discrimination |
|---|---|---|---|
| **Signal-to-noise** | Gu–Kelly–Xiu (2020), Giglio–Kelly–Xiu (2022) | Rank IC and conditional loss by month move with cross-sectional dispersion; higher common shocks may obscure characteristic signals | Cross-sectional dispersion and conditional realized idiosyncratic/common volatility; avoid mechanically correlated denominators; estimate uncertainty around slopes |
| **Attention/information processing** | Momentum evidence in Jegadeesh–Titman (1993); more direct attention papers should be searched before specific claims | Conditional momentum and value interactions appear strongest where information diffuses slowly | Could reflect compensation for risk or shorting constraints, not underreaction; add defensible dated investor-attention proxies |
| **Funding liquidity/limits to arbitrage** | Avramov–Cheng–Metzker (2023); McLean–Pontiff (2016); Novy-Marx–Velikov (2016) | EW signal weakens sharply after microcap/illiquidity screens and borrowing/impact sensitivity | Risk exposures, trading capacity and data microstructure artifacts; compare EW with VW and explicit investability filters |
| **Time-varying risk exposures** | Kelly–Pruitt–Su (2019); Fama–French (2015); Moreira–Muir (2017); Bekaert–Hoerova (2014) | Gross returns disappear after conditional factor-risk adjustment or characteristic beta controls | Mispricing versus priced risks cannot be inferred by VIX-state differences alone; conduct exposure tests and compare alternatives |
| **Nonlinear interactions** | Gu–Kelly–Xiu (2020); Freyberger–Neuhierl–Weber (2020); Chen–Pelger–Zhu (2024) | XGBoost−Elastic Net paired ranking/loss differences survive under common predictor sets; partial-dependence structures stable through subperiods | Overfitting, sample composition, mere sorting artifacts; validate with ablation, stable SHAP/permutation caveats, benchmarks |
| **Momentum crash exposure** | Daniel–Moskowitz (2016); Barroso–Santa-Clara (2015) | Negative performance clusters in crash/panic months or sharply rebounding markets | General VIX channel vs known nonlinear momentum crash risk; control for past market drawdown and realized rebound without look-ahead |

## Identification discipline and M17 checklist
1. Define proxy data source and calendar availability before extracting or modeling.
2. Report descriptive conditional associations, confidence intervals and multiple-test inventory; no exogenous assignment is available.
3. Do not claim mechanisms can be distinguished with only current three characteristics and VIX; additional observables and factor exposure controls are needed.
4. Compare predictive signal quality (IC, forecast loss) separately from gross/net implementation outcomes.
5. If no mechanism discriminates robustly, explain alternative interpretations and stop short of causal inference.

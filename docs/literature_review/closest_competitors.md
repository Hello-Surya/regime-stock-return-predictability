# Closest-Competitor Audit — Literature as of 2026-10-20

## Direct and adjacent studies
This comparison is based on verified abstracts and publisher metadata; features not established from accessible text are marked **unverified**, not assumed absent. Use the full-text audit to settle each row before a claim of priority.

| Study | Individual-stock prediction | Volatility/state conditioning | Nonlinear vs linear | Economic/turnover question | Level of differentiation |
|---|---|---|---|---|---|
| [Gu, Kelly, Xiu 2020](https://doi.org/10.1093/rfs/hhaa009) | Yes | Aggregate-variable interactions | Yes | Economic ML gains | **Moderate**: our restricted three-predictor tercile/HAC framing versus their broad ML comparison |
| [Avramov, Cheng, Metzker 2023](https://doi.org/10.1287/mnsc.2022.4449) | Yes | High market-volatility and limits-to-arbitrage states | Deep learning and economically restricted alternatives | Microcaps, turnover, costs | **Limited**: direct conceptual overlap with our critical findings |
| [Chen, Pelger, Zhu 2024](https://doi.org/10.1287/mnsc.2023.4695) | Yes | Learned macroeconomic states | Deep conditional asset-pricing | Sharpe / pricing errors | **Moderate**: different estimand and ex-ante VIX state definition |
| [Freyberger, Neuhierl, Weber 2020](https://doi.org/10.1093/rfs/hhz123) | Yes | Unverified | Nonparametric vs linear | OOS forecast claims | **Moderate**: general nonlinear characteristic insights precede us |
| [Lewellen 2015](https://doi.org/10.1561/104.00000024) | Yes | Unverified | Characteristic regressions | Predictive slopes | **Moderate**: natural stronger simple benchmark, not yet implemented here |
| [Murray, Xia, Xiao 2024](https://doi.org/10.1016/j.jfineco.2024.103791) | Yes | Unverified | ML on chart features | Evidence among large stocks | **Moderate**: different signals, challenges broad claims about VW failure |
| [Hou, Xue, Zhang 2020](https://doi.org/10.1093/rfs/hhy131) | Anomaly portfolios | Not established | No main ML comparison | Microcaps, VW, testing | **Moderate**: provides investability yardstick not new hypothesis |
| [Novy-Marx, Velikov 2016](https://doi.org/10.1093/rfs/hhv063) | Anomaly portfolios | Not established | Not central | Turnover and transaction costs | **Moderate**: net evaluation alone is not novel |
| [Daniel, Moskowitz 2016](https://doi.org/10.1016/j.jfineco.2015.12.002) | Momentum-related portfolios | Panic/high volatility | Not same ML comparison | Conditional momentum crashes | **Moderate**: important mechanism confounder |
| [Jensen, Kelly, Malamud, Pedersen 2026](https://doi.org/10.1093/rfs/hhag022) | Yes, individual stock characteristics for portfolio construction | No VIX-tercile design in examined setup | Cost-aware Portfolio-ML against standard ML optimizers | Explicit cost-aware frontier, capacity, net performance | **Limited** differentiation for integrated ML/cost narrative; **moderate** for VIX conditional IC specifically |
| [Fang, Ślepaczuk 2026, SSRN working paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6898918) | No — high-frequency Chinese CSI 300 index | Markov-switching and volatility forecasting | XGBoost return prediction | Walk-forward and cost-sensitive evaluation | **Moderate task differentiation**: extremely related *question*, but different asset, frequency and working-paper status |
| [Bekaert, Hoerova 2014](https://doi.org/10.1016/j.jeconom.2014.05.008) | No — aggregate market | VIX decomposition | Different econometrics | Aggregate return forecasts | **Strong task differentiation** but VIX motivation is established |

## Referee-facing conclusions
**Additional current challenge:** Jensen–Kelly–Malamud–Pedersen (2026, RFS) already shows that direct portfolio optimization with trading costs changes the economic merits of ML signals. Treat simple 50-bp net deciles as a preliminary benchmark rather than a methodological first.

**Closest intellectual challenge:** Avramov–Cheng–Metzker already find that deep-learning profits are linked to hard-to-arbitrage stocks, and attenuate with high volatility, microcap restrictions and costs. Do not claim that the conjunction “ML + volatility + costs” or “EW positive, VW weak” is original.

**Possible bounded contribution:** a pre-visible-information expanding VIX-tercile *evaluation* (not regime-trained models), comparing monthly Spearman IC, forecast-loss differences, formal simultaneous regime contrasts, and identical frozen forecasts' 0/50-bp EW/VW return streams. This becomes meaningful only with strong benchmark, regime-threshold, subperiod, liquidity, factor-risk and alternative-specification evidence.

**No direct-priority claim:** A June 2026 non-refereed paper (Fang–Ślepaczuk) explicitly studies regime-aware XGBoost return prediction, low-volatility concentration and transaction costs for a high-frequency Chinese equity index. The working-paper status and different target should be stated, not ignored. The present search does **not** establish that no earlier paper has the precise same evaluation. “First”, “novel ML methodology”, and “causal influence of VIX” must not appear in the manuscript.

## Full-text follow-up per competitor
For each of the ten studies above, record sample dates/frequency, predictors, timing, regimes, OOS design, exact model and inference procedures, cost assumptions, core estimand, working-paper history and table numbers. The present grid checks only assertions explicit in accessible abstracts and metadata; blank/unverified cells are not negative findings.

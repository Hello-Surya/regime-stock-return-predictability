# Research Claims and Evidence Register (as of 2026-10-20)

| ID | Proposed manuscript statement | Category / status | Literature | Existing local-project evidence | Missing evidence / counterargument | Publication status |
|---|---|---|---|---|---|---|
| C01 | Nonlinear models are already established tools for stock-return prediction | External literature: established | Gu–Kelly–Xiu 2020; Freyberger et al. 2020 | Baseline EN/XGB code | Not a claim of our novelty | Cite accurately |
| C02 | Volatility is associated with aggregate expected-return risk and time-varying risk premia | External literature: established but aggregate targets differ | Bekaert–Hoerova 2014; Bollerslev et al. 2009 | Formation-time VIX regime labels | Aggregate findings not stock-level effects | Cite with scope |
| C03 | Overall stock-level out-of-sample R² is near zero | Supported by reported project analysis | Not required | \`README.md\`, \`paper/sections/04_results.tex\` | Fresh WRDS reproduction | Report as project result, not literature generality |
| C04 | XGBoost LOW *binary* Spearman IC ≈0.01867; p≈0.013 | Supported by reported project analysis | M9 HAC concept Newey–West 1987 | \`paper/sections/04_results.tex\` | Dependence/holdout sensitivity | Report with binary definition and conditional nature |
| C05 | Direct LOW/HIGH binary rank-IC difference is established | Contradicted/unsupported | Statistical contrast framework | p≈0.105 | Large CI and Holm/other families | **Do not claim** |
| C06 | XGB EW LOW net three-state mean ≈0.6135%/month under assumed 50 bp | Supported by reported project analysis | Novy-Marx–Velikov 2016 for why costs matter | \`results/transaction_costs/transaction_cost_summary.md\`, p=.0143 | Actual costs, data reproduction, dependence alternatives | Can report as *modeled* net point estimate |
| C07 | Net returns differ robustly across three regimes | Contradicted/unsupported | Holm 1979 | For XGB EW HIGH−MIDDLE raw p=.0462, Holm p=.1386; all three-state adjusted p>0.05 | Greater precision/independent sample | **Do not claim** |
| C08 | XGB EW signal survives value weighting | Contradicted/unsupported | Hou–Xue–Zhang 2020 | XGB VW net LOW −0.0820%, MIDDLE −0.1952%, HIGH −0.4717% | Alternate size universes | **Do not claim** |
| C09 | Positive modeled net returns establish investable profit | Unsupported | Avramov–Cheng–Metzker 2023; Novy-Marx–Velikov 2016; Jensen et al. 2026 | 50-bp proportional trade costs only | Borrow/locate/funding, impact/capacity | **Do not claim** |
| C10 | Results are driven by microcaps and limits to arbitrage | Tentative explanation | Avramov–Cheng–Metzker 2023; Hou–Xue–Zhang 2020 | EW–VW gap | Actual size/liquidity attribution | Untested mechanistic statement |
| C11 | Three-state middle-volatility peak is reproducible | Exploratory/post-hoc | No definitive direct prior known from abstract search | MIDDLE larger EW descriptive mean | Holm/threshold sensitivity, independent holdout | Hypothesis only |
| C12 | VIX causally changes return predictability | Unsupported | No identification | Observational state split | IV/shock identification absent | **Do not claim** |
| C13 | XGBoost has positive incremental predictive advantage over EN in every state | Untested as paired comparison | Gu–Kelly–Xiu 2020 | Separate model IC/portfolio tables | M12 aligned IC and loss model contrasts | Not established |
| C14 | Exact three-state net evaluation is a unique first contribution | Not demonstrated | Avramov–Cheng–Metzker 2023 overlaps; new studies may exist | Combining components in this code | Full-text competitor audit; novel substantive result | **No priority claim** |
| C15 | The study has a transparent timing-respecting infrastructure | Supported by committed design/documentation, not independent empirical audit | Leakage studies and forecast literature | \`03_methods.tex\`, production code, tests | External replication and fresh build | Methods claim with explicit caveat |
| C16 | A conditional null finding could be informative | Tentative scientific positioning | McLean–Pontiff 2016; Hou–Xue–Zhang 2020 | Inconclusive contrasts and VW failure | Power and strong benchmark comparisons | Avoid claiming journal contribution prematurely |

Statuses are intentionally not binary; **documented code** is not **independent validation** and **non-rejection** is not proof of no effect.

Before each paper revision, map every assertive sentence to a row, cite its basis, update result provenance and validate numbers. Null and failed robustness outcomes remain in the record.

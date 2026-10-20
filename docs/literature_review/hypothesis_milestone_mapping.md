# Research Questions / Hypotheses to Milestones 12–18

Status classification: M9/M10 are already reported. M12–18 are **future** analyses only; none is launched by this documentation.

| ID | Literature base | Existing evidence at M11 | Missing work | Future milestone | Expected artifact | Scientific risk and validation |
|---|---|---|---|---|---|---|
| RQ1/H1 regime-dependent cross-sectional IC | Gu–Kelly–Xiu 2020; Chen–Pelger–Zhu 2024; Bekaert–Hoerova 2014 | LOW positive IC, descriptive MIDDLE and HIGH; gross M9 HAC and three-state contrasts | Joint HAC, bootstrap blocks, alternative historical cutoffs and rolling windows | 14,16 | \`results/robustness/regime_ic_joint_tests.csv\`; paper conditional inference table | No adjusted direct differences; date alignment, no look-ahead, adjusted families |
| RQ2/H2 relative XGB vs EN | Gu–Kelly–Xiu 2020; Freyberger–Neuhierl–Weber 2020; Diebold–Mariano 1995; Giacomini–White 2006 | Both original models evaluated separately with same three predictors | Monthly paired \`IC_XGB-IC_EN\`; signed same-stock monthly loss differential; historical mean/OLS/characteristic benchmark tests; valid conditional comparison | 12,16 | \`results/benchmarks/paired_model_tests.csv\`; manuscript benchmark comparison | Selection and error dependence; enforce identical months/stocks, benchmark-specific inference |
| RQ3/H3 economic consequences | Avramov–Cheng–Metzker 2023; Novy-Marx–Velikov 2016; Jensen et al. 2026 | Net LOW EW XGB +0.6135% in three-state M10, VW negative; Holm pairwise not significant | Stronger trading frictions, financing, borrow, risk-adjusted net returns and size restrictions | 15,17 | \`results/investability/net_portfolios_by_screen.csv\`, \`results/factor_alpha/net_alpha.csv\` | Proxy implementation costs, capacity, significance vs tradability |
| RQ4/H4 intermediate-state peak (exploratory) | Daniel–Moskowitz 2016; Bekaert–Hoerova 2014 | MIDDLE EW gross/net largest descriptively, but observed post hoc | Preset intersection-union/simultaneous CI, threshold sensitivity, genuinely separate sample | 14,16 | \`results/robustness/intermediate_peak.csv\` | Data snooping, multiple candidates, short regime sequences; label exploratory |
| RQ5/H5 stability/investability | Hou–Xue–Zhang 2020; McLean–Pontiff 2016; DeMiguel et al. 2020 | VW erases EW signal; 50-bp assumptions inadequate for some names | Investable universe, size/breakpoint and volume restrictions, subperiod holdout, borrow/impact scenarios | 15,16 | \`results/investability/screen_flow.csv\`, portfolio figure, robustness appendix | Sample composition and survivor effects; document each selection change |
| RQ6 mechanisms/interpretation | Kelly–Pruitt–Su 2019; Moreira–Muir 2017; Avramov et al. 2023 | Observational VIX patterns only | Factor exposures; importance/ablation; liquidity, attention, momentum crash controls | 13,17 | \`results/mechanisms/factor_exposures.csv\`, explanation plots | Correlated channels; no causal identification from regime labels |
| Overall journal contribution | Gu et al.; Avramov et al.; cross-sectional literature | Documented transparent pipeline and mixed empirical results | Integrate verified closest competitor full-text analysis, selected exercises and audit | 18 | Complete paper and replication package | No demonstrated differentiation if current patterns vanish and existing work already covers question |

## Phase gates
**M12:** Declare monthly evaluation unit, loss sign and benchmark grid *before* computing new tests; do not change original forecasts.  
**M13:** Record added features separately; frozen three-characteristic model remains baseline.  
**M14:** Evaluate alternative VIX definitions as robustness, not retrospectively tuned signal optimizers.  
**M15:** Protect M8 portfolio identities for M10 replication but construct *new labeled investability universes* with screened results separate.  
**M16:** Address serial dependence and known subperiod shifts; frozen original sample remains archival.  
**M17:** Distinguish prediction interpretation from causation.  
**M18:** Re-run audit from clean environment, validate citations/full texts, factor/risk tables and manuscript claims. No selective omission of null results.

All path names in this mapping are **planned output suggestions**, not files that currently exist.

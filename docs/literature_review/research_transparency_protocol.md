# Research Transparency Protocol — Retrospective vs Future Analysis

**Protocol written 2026-10-20, after production results were inspected. Not a preregistration.** It cannot create an untouched evaluation period inside the already-studied January 2000–November 2025 sample.

## Recorded timeline and origins
- **Prior to Milestone 11:** Original project question on conditional cross-sectional stock forecasting; size, B/M, momentum predictors; Elastic Net and XGBoost; strict expanding window and original expanding-median LOW/HIGH VIX split were chosen and evaluated. Baseline inference and implementation already produced outcome tables.
- **Milestone 8:** LOW/MIDDLE/HIGH expanding-tercile evaluation was appended to the original binary split. Middle-state ranking/return values were observed before this protocol.
- **Milestone 9:** Three-state monthly HAC tests and Holm adjustment; direct regime differences not Holm-significant.
- **Milestone 10:** Costs and transaction turnover on frozen portfolios. LOW three-state EW XGBoost net result positive, VW net negative, pairwise adjusted differences inconclusive.
- **Milestone 11:** Current contribution, hypotheses, mechanisms and future robustness analyses documented with hindsight. H4 “middle outperforming both extremes” is explicitly **data-informed exploratory**.
- **Milestones 12–18:** Future calculations must be logged before execution. They may be design-locked going forward, but that does not make the 2000–2025 sample independent.

## Existing empirical outcomes and their roles
1. **Ranking:** monthly Spearman cross-sectional information coefficient (IC); descriptive and inferential by regime.
2. **Economic outcomes:** monthly D10–D1 EW/VW gross and net after *assumed* cost scenarios.
3. **Forecast loss:** RMSE/MAE/OOS R² on original frozen panel; forthcoming fair paired-loss comparisons.
4. **Direct conditional differences:** HAC joint regime contrasts and Holm-adjusted pairwise tests preferred over within-state significance comparisons.
5. **Exploratory outcomes:** intermediate-volatility maximum, alternative cutoff sensitivity and mechanism proxies.

Do not elevate a retrospective outcome to prospective primary merely by adding it to a dated paper section. If the analysis uses a newly gathered future non-overlapping period or independent external sample, describe its construction, information cutoff and additional limitations; a later re-test on previously observed timestamps is not an untouched test.

## Testing register for every M12–17 run
Before reading new outcomes, record: (i) date + branch/commit and author; (ii) precise null, alternative and outcome; (iii) forecast/realization formation information cutoff; (iv) sample date range and state labels; (v) model, predictors, weights, cost rates, eligibility; (vi) test statistic and dependence estimator; (vii) multiple-testing family and adjustment method; (viii) planned negative-result reporting; (ix) code/config SHA and output paths; (x) any exploratory deviations and why they occurred.

Freeze comparison families within appropriate outcome and model/weight combinations before testing. For multiple families/models/regime definitions, report all run specifications, including adjusted and raw p-values and common confidence intervals where estimable. Do not select bandwidth, cost scenario, threshold, subperiod, model seed or predictor based on maximal significance without treating it as exploratory.

## Practical integrity controls
- Preserve archived M7 predictions, M8 portfolios, M9 gross HAC tables and M10 original cost assumptions; do not overwrite or silently re-estimate them during M11.
- Keep WRDS raw row-level CRSP/Compustat datasets, secrets, detailed private holdings out of Git.
- Use signed/difference contrasts and confidence intervals, not a significant-state vs insignificant-state argument.
- Distinguish binary expanding-median from three-state expanding-tercile tables everywhere.
- Include negative, null, unstable and economically unimplementable estimates in final paper and appendices.
- When a hypothesis or result is changed, log its earlier wording, motivation and full commit history.
- Publish limitations: short-side borrow costs, funding, price impact, microcaps, benchmark omission and observational VIX confounding.

## Mandatory sign-off
Faculty/advisor review of closest competing full texts and hypotheses; check for overlap with Avramov–Cheng–Metzker 2023 before making journal novelty statements; test clean branch in local licensed-data environment; audit LaTeX and citations from clean build; annotate tests actually performed. A software test suite cannot certify truth of a citation, novelty or causal interpretation.

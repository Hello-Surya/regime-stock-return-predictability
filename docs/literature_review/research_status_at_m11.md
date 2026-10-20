# Verified Research Status at Milestone 11
Audit: 2026-10-20. Source: GitHub `main`, latest inspected commit `f89d8970ef794eba83cfca5bc2fef87c795320ba` (2026-10-20), plus manuscript, README, pipeline and summary files. This is a remote repository audit, **not** a reproduction of local licensed-data outputs.

## Stages and evidence
| Stage | Repository evidence | Status | Important qualification |
|---|---|---|---|
| M1–M7 data construction and expanding OOS models | `docs/data_sources.md`, `docs/baseline_model_estimation.md`, `paper/sections/03_methods.tex`, `paper/sections/04_results.tex` | Implemented, documented, reported production | Original CRSP/WRDS row-level inputs local; not independently reproduced here |
| M8 binary and three-state portfolio evaluation | `paper/sections/05_economic_value.tex`, `results/economic_value/` documented in README | Implemented, reported | Three-state mapping describes **evaluation of frozen forecasts**, not three regime-specific models |
| M9 HAC rank/return inference, regime differences, Holm adjustments | Commit `2021382f4c2dd12370f3025e8000860a9dca4fcb`, results section and `src/rdsrp/inference` | Implemented, committed, empirically reported | No three-state pairwise XGBoost return/IC contrast survives Holm at 5% |
| M10 turnover, 0/50-bp net costs | Commit `f89d8970ef794eba83cfca5bc2fef87c795320ba`, `src/rdsrp/transaction_costs/pipeline.py`, `tests/test_transaction_costs.py`, `paper/generated/table_transaction_costs.tex`, `results/transaction_costs/transaction_cost_summary.md` | Implemented, committed, production summary reported | Tests are synthetic and are **not** independent row-level WRDS regeneration |

## Frozen design
US common equity, NYSE/AMEX/Nasdaq (production uses current CRSP CIZ classification rather than merely legacy share codes); $5 *formation-month* minimum price; monthly 1990–2025 panel with June book-equity convention; 1-month-ahead individual-stock return; log size, book-to-market and 12–2 momentum; Elastic Net versus 200-tree depth-2 XGBoost; 1990–1999 historical forward-chaining hyperparameter selection followed by monthly expanding-window refits; 2000-01 to 2025-11 OOS. VIX observed at formation date; original expanding-median binary regime is unchanged; later expanding-tercile evaluation yields three states.

## Baseline empirical sample and outcomes
2,140,091 full panel rows; 1,798,891 complete baseline predictors; 1,340,823 final screened panel rows; 893,486 OOS stock-month predictions over 311 formation months. Overall OOS R² is approximately -0.00027 (Elastic Net) and -0.00023 (XGBoost). Original binary LOW-VIX XGBoost Spearman IC = 0.01867 (p=0.0132); binary LOW/HIGH difference not significant (p≈0.105).

Binary LOW XGBoost equal-weighted net long–short = 0.748% per month under one-way 50-bp turnover, HAC p=0.0049, but value-weighted net negative. These **binary** results are not interchangeable with the **three-state** numbers.

Three-state M10 XGBoost equal-weighted net monthly D10–D1: LOW 0.6135% (p=0.0143), MIDDLE 0.9601% (p=0.0714), HIGH -0.5152% (p=0.4388). XGBoost value-weighted net: LOW -0.0820%, MIDDLE -0.1952%, HIGH -0.4717%. Net HIGH–MIDDLE EW difference = -1.475 percentage points, raw p=0.0462 but Holm p=0.1386. None of the XGBoost three-state pairwise adjusted return contrasts is significant at 5%. Baseline cost is an assumed proportional model without explicit borrow fees, funding costs or nonlinear impact.

## Implemented / tested / empirically validated / documented
- **Implemented and committed:** source code, static tables, paper text, transaction-cost report.
- **Tests present:** synthetic scientific/accounting tests in `tests/test_transaction_costs.py`; remote execution not performed in this audit.
- **Empirically reported:** production aggregates in the tracked summary, figures/table; **not independently reproduced in this environment**.
- **Documented:** methodology, data rules, inference, exact sample and reported estimates.
- **Proposed:** independent robustness, comparison with historical-mean and characteristic benchmarks, alternative VIX thresholds, liquidity and microcap exclusions, subperiod analysis, interpretability and full referee audit (M12–18).
- **Incomplete at M11 start:** publication literature context, bibliography, fully specified hypotheses; independent full-text paper-review, regression reproduction, LaTeX build and current local Git status **cannot be certified remotely**.

## Required cautions
1. *Significant in LOW but not HIGH* is **not** a significant LOW–HIGH difference.
2. *Three-state gross* inference is not automatically net cost-adjusted inference.
3. Binary and tercile regimes produce different samples and means; never compare without labels.
4. EW success with VW failure may reflect liquidity, microcap exposure or other omitted mechanisms; not proof of any particular cause.
5. Treat model forecasts as associations, not causal VIX effects.
6. The local working tree may have uncommitted changes not visible in GitHub; changes here are isolated in `milestone-11-literature-and-hypotheses`.

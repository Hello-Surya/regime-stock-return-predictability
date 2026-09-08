# Repository Audit — Milestone 1

## Initial state

The repository already had a sound research-oriented layout: `src/rdsrp/` separated data, features, regimes, models, evaluation, portfolios, paper helpers, and utilities; `scripts/`, `configs/`, `tests/`, `docs/`, `results/`, and `paper/` were also present.

## Functionality found

- YAML files encoded the intended monthly common-stock universe, one-month-ahead target, VIX expanding-median regime rule, Elastic Net/XGBoost model set, expanding-window OOS evaluation, and portfolio design.
- Configuration loading, logging scaffolding, model interfaces, and module boundaries existed.
- The test directory contained smoke-test placeholders.

## Missing or non-executable components found

- `scripts/run_pipeline.py` terminated intentionally rather than executing a pipeline.
- Feature construction, VIX regime labeling, expanding-window splits, model fitting, OOS prediction, cross-sectional transforms, and most research artifact functions were stubs.
- Elastic Net and XGBoost returned placeholder zero predictions instead of fitted model outputs.
- Existing tests mainly verified empty/stub behavior rather than econometric timing rules.
- The WRDS package was not declared in the original project dependencies.

## Methodological risks identified

- Target shifting needed to occur within security identifiers to avoid cross-security contamination.
- Momentum needed an explicit 12–2 implementation excluding months t and t-1.
- The VIX threshold needed an expanding calculation that cannot change historical classifications when future VIX values are added.
- Expanding-window training needed a strict `train feature date < prediction feature date` rule so the target attached to the final training month is already realized.
- Standardization for Elastic Net needed to be fitted inside each historical training sample only.

## Milestone 1 implementation path

Milestone 1 therefore implements a complete synthetic software-validation path while preserving the repository architecture. It adds leakage-aware feature construction, expanding VIX regimes, real Elastic Net/XGBoost estimators, expanding-window OOS predictions, metrics, plots, validation reports, Windows-oriented documentation, and timing/leakage tests. Synthetic outputs are explicitly isolated from empirical research outputs.

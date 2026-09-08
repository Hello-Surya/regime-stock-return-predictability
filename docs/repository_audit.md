# Repository Audit

The repository began with a sound research-oriented architecture separating data, features, regimes, models, evaluation, portfolios, paper helpers, scripts, configurations, tests, documentation, and results.

The initial audit found that feature construction, VIX regime labeling, expanding-window splits, model fitting, OOS prediction, cross-sectional transforms, and most research artifact functions were stubs. Elastic Net and XGBoost returned placeholder outputs rather than fitted predictions.

The software foundation now includes leakage-aware feature construction, expanding VIX regimes, real Elastic Net/XGBoost estimators, expanding-window OOS predictions, metrics, plots, validation reports, Windows-oriented execution documentation, and timing/leakage tests.

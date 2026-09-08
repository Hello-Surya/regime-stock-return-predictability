"""High-level research pipeline orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from rdsrp.data.cache import read_parquet_cache
from rdsrp.data.synthetic import make_synthetic_market
from rdsrp.data.vix import cached_vix_daily, vix_daily_to_monthly
from rdsrp.data.wrds import cached_crsp_monthly, connect_wrds
from rdsrp.eval.metrics import metrics_table
from rdsrp.eval.oos import run_expanding_oos
from rdsrp.features.build import build_features
from rdsrp.models.elastic_net import ElasticNetModel
from rdsrp.models.tuning import select_params_time_cv
from rdsrp.models.xgboost_model import XGBoostModel
from rdsrp.paper.figures import plot_actual_vs_predicted, plot_regime_performance, plot_vix_regimes
from rdsrp.paper.tables import model_comparison
from rdsrp.regimes.vix_regime import label_vix_regime
from rdsrp.utils.checks import data_validation_report, validate_oos_predictions
from rdsrp.utils.io import ensure_dir
from rdsrp.utils.serialization import write_json


def _wide_stock_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "ticker": ("ticker", "first"),
        "vix": ("vix", "first"),
        "regime": ("regime", "first"),
        "actual_next_month_return": ("actual", "first"),
    }
    if "benchmark_prediction" in predictions:
        aggregations["historical_mean_benchmark"] = ("benchmark_prediction", "first")
    base = predictions.groupby("date", as_index=False).agg(**aggregations)
    wide = predictions.pivot_table(index="date", columns="model", values="prediction", aggfunc="first").reset_index()
    wide = wide.rename(columns={"elastic_net": "elastic_net_prediction", "xgboost": "xgboost_prediction"})
    return base.merge(wide, on="date", how="left").sort_values("date").reset_index(drop=True)


def _metric_line(frame: pd.DataFrame, model: str) -> str:
    if model not in frame.index:
        return "insufficient observations"
    row = frame.loc[model]
    return (
        f"N={int(row['n_obs'])}, RMSE={row['rmse']:.6f}, MAE={row['mae']:.6f}, "
        f"R2={row['r2']:.4f}, OOS R2={row['oos_r2']:.4f}, "
        f"correlation={row['correlation']:.4f}"
    )


def _write_synthetic_summary(output_dir: Path, selected_ticker: str, stock_predictions: pd.DataFrame, metrics: pd.DataFrame, feature_cols: list[str]) -> None:
    overall = metrics[metrics["regime"] == "OVERALL"].set_index("model")
    high = metrics[metrics["regime"] == "HIGH"].set_index("model")
    low = metrics[metrics["regime"] == "LOW"].set_index("model")
    sample_start = pd.to_datetime(stock_predictions["date"]).min().date()
    sample_end = pd.to_datetime(stock_predictions["date"]).max().date()
    text = f"""# Regime-Dependent Stock Return Predictability — Synthetic Software Validation Results

> **Synthetic validation only. These values are generated data and are not empirical research findings.**

## Sample

The software-validation run produces {stock_predictions['date'].nunique()} out-of-sample prediction months for {selected_ticker}, spanning {sample_start} to {sample_end}. Models are trained on a larger synthetic cross-section, while this report displays the selected stock's final OOS predictions.

## Selected stock

Selected synthetic ticker: **{selected_ticker}**.

## Predictor set

The validation pipeline uses: {', '.join(feature_cols)}. Predictor construction is attached to feature month t; the target is the return at t+1.

## Regime definition

The regime is HIGH when monthly VIX at t is greater than the expanding historical median calculated using observations through t; otherwise it is LOW. No future VIX observations enter classification.

## Elastic Net performance

Overall: {_metric_line(overall, 'elastic_net')}.

## XGBoost performance

Overall: {_metric_line(overall, 'xgboost')}.

## High-volatility results

Elastic Net: {_metric_line(high, 'elastic_net')}.

XGBoost: {_metric_line(high, 'xgboost')}.

## Low-volatility results

Elastic Net: {_metric_line(low, 'elastic_net')}.

XGBoost: {_metric_line(low, 'xgboost')}.

## Interpretation

This run establishes software operability only. The numerical performance has no empirical interpretation because the input data are synthetic.

## Current limitations

- No WRDS observations are used in synthetic validation.
- Synthetic book-to-market exists only to exercise the software path.
- Production Compustat/CCM accounting-data timing is not represented here.
"""
    (output_dir / "preliminary_results_summary.md").write_text(text, encoding="utf-8")


def _write_empirical_summary(output_dir: Path, selected_ticker: str, stock_predictions: pd.DataFrame, metrics: pd.DataFrame, feature_cols: list[str], config: dict[str, Any], metadata: dict[str, Any]) -> None:
    overall = metrics[metrics["regime"] == "OVERALL"].set_index("model")
    high = metrics[metrics["regime"] == "HIGH"].set_index("model")
    low = metrics[metrics["regime"] == "LOW"].set_index("model")
    sample_start = pd.to_datetime(stock_predictions["date"]).min().date()
    sample_end = pd.to_datetime(stock_predictions["date"]).max().date()
    real_cfg = config["run"]["real"]
    text = f"""# Regime-Dependent Stock Return Predictability — Preliminary Empirical Results

## Sample

This is a genuine WRDS-based preliminary empirical run. The displayed stock has {stock_predictions['date'].nunique()} out-of-sample prediction months spanning {sample_start} to {sample_end}. Models are trained on the historical reduced cross-section rather than on {selected_ticker} alone.

CRSP source: **{metadata.get('crsp_source', 'local cache')}**. The quick-run cross-section uses a fixed universe formed from the top {real_cfg['top_n_by_market_cap']} eligible common stocks by market equity on {real_cfg['universe_formation_date']}, with the selected ticker retained explicitly. The ${real_cfg['min_price']:.0f} price screen is applied at feature month t after full return histories are pulled, so a future price decline cannot make the t+1 target disappear.

## Selected stock

Selected common stock: **{selected_ticker}**.

## Predictor set

**PRELIMINARY PARTIAL FEATURE SET:** {', '.join(feature_cols)}. Book-to-market is intentionally not fabricated; it will enter after Compustat/CCM construction. The target is the next-calendar-month CRSP return.

## Regime definition

Daily Cboe VIX data are converted to the final observed VIX close in each calendar month. HIGH means VIX at month t exceeds the expanding historical median using only VIX observations through t; otherwise the regime is LOW. The VIX median history begins at {real_cfg['vix_history_start']}.

## Elastic Net performance

Overall: {_metric_line(overall, 'elastic_net')}.

## XGBoost performance

Overall: {_metric_line(overall, 'xgboost')}.

## High-volatility results

Elastic Net: {_metric_line(high, 'elastic_net')}.

XGBoost: {_metric_line(high, 'xgboost')}.

## Low-volatility results

Elastic Net: {_metric_line(low, 'elastic_net')}.

XGBoost: {_metric_line(low, 'xgboost')}.

## Interpretation

OOS R2 is computed against a stock-specific expanding historical-mean return benchmark constructed from information available before each prediction month. Positive values indicate improvement over that benchmark; negative values indicate that the historical-mean benchmark performs better. No significance claims are made at this stage.

## Current limitations

- This is a computationally reduced preliminary run using a fixed start-of-sample large-cap universe, not the full CRSP production universe.
- The predictor set currently contains size and momentum only; book-to-market awaits the Compustat/CCM stage.
- Hyperparameters are selected once using forward-chaining validation entirely inside the initial historical training period, then frozen for the preliminary OOS run.
- Models are refit every {real_cfg['retrain_every_months']} months in quick mode rather than every month.
- Delisting-return treatment and the complete production universe are finalized in the complete-data milestone.
- Portfolio sorts, transaction costs, and formal inference are later milestones.
"""
    (output_dir / "preliminary_results_summary.md").write_text(text, encoding="utf-8")


def run_synthetic_validation(repo_root: Path, ticker: str = "STK001", n_stocks: int = 40, n_months: int = 54, oos_months: int = 12, random_state: int = 42) -> dict[str, Path]:
    output_dir = ensure_dir(repo_root / "results" / "synthetic_validation")
    for old in output_dir.iterdir():
        if old.is_file() and old.name != ".gitkeep":
            old.unlink()
    crsp, vix = make_synthetic_market(n_stocks=n_stocks, n_months=n_months, random_state=random_state)
    features = build_features(crsp)
    regimes = label_vix_regime(vix)
    panel = features.merge(regimes[["date", "vix", "vix_expanding_median", "regime"]], on="date", how="left", validate="many_to_one")
    feature_cols = ["log_me", "book_to_market", "mom_12_2"]
    data_validation_report(panel, feature_cols).to_csv(output_dir / "data_validation.csv", index=False)
    unique_dates = sorted(panel["date"].dropna().unique())
    if oos_months >= len(unique_dates):
        raise ValueError("oos_months must be smaller than the total synthetic sample.")
    start_test = pd.Timestamp(unique_dates[-oos_months])
    models = {
        "elastic_net": ElasticNetModel(params={"alpha": 0.001, "l1_ratio": 0.5, "max_iter": 10000}, random_state=random_state),
        "xgboost": XGBoostModel(params={"n_estimators": 60, "max_depth": 2, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "n_jobs": 1}, random_state=random_state),
    }
    predictions = run_expanding_oos(panel, feature_cols=feature_cols, models=models, start_test=start_test, end_test=panel["date"].max(), min_train_months=18)
    validate_oos_predictions(predictions)
    stock = predictions[predictions["ticker"].str.upper() == ticker.upper()].copy()
    if stock.empty:
        available = ", ".join(sorted(predictions["ticker"].dropna().unique())[:8])
        raise ValueError(f"Ticker '{ticker}' is not in synthetic sample. Examples: {available}")
    stock.to_csv(output_dir / "single_stock_predictions_long.csv", index=False)
    _wide_stock_predictions(stock).to_csv(output_dir / "single_stock_predictions.csv", index=False)
    metrics = metrics_table(stock)
    metrics[metrics["regime"] == "OVERALL"].to_csv(output_dir / "single_stock_metrics.csv", index=False)
    metrics[metrics["regime"].isin(["HIGH", "LOW"])].to_csv(output_dir / "regime_metrics.csv", index=False)
    model_comparison(metrics).to_csv(output_dir / "model_comparison.csv", index=False)
    plot_actual_vs_predicted(stock, output_dir / "actual_vs_predicted.png")
    plot_vix_regimes(regimes, output_dir / "vix_regimes.png")
    plot_regime_performance(metrics, output_dir / "regime_performance.png")
    _write_synthetic_summary(output_dir, ticker.upper(), stock, metrics, feature_cols)
    return {"output_dir": output_dir, "predictions": output_dir / "single_stock_predictions.csv", "metrics": output_dir / "single_stock_metrics.csv", "summary": output_dir / "preliminary_results_summary.md"}


def _load_run_config(repo_root: Path, quick: bool) -> dict[str, Any]:
    path = repo_root / "configs" / ("quick.yaml" if quick else "production.yaml")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Run configuration is invalid: {path}")
    return config


def run_wrds_single_stock(repo_root: Path, ticker: str = "AAPL", *, quick: bool = True, refresh: bool = False) -> dict[str, Path]:
    config = _load_run_config(repo_root, quick=quick)
    run_cfg = config["run"]
    real_cfg = run_cfg["real"]
    random_state = int(run_cfg.get("random_state", 42))
    output_dir = ensure_dir(repo_root / "results" / "preliminary")
    for old in output_dir.iterdir():
        if old.is_file() and old.name != ".gitkeep":
            old.unlink()
    cache_tag = f"{real_cfg['crsp_start'][:4]}_{real_cfg['crsp_end'][:4]}_fixedtop{real_cfg['top_n_by_market_cap']}_p{int(real_cfg['min_price'])}"
    crsp_cache = repo_root / "data" / "raw" / f"crsp_monthly_{cache_tag}.parquet"
    vix_cache = repo_root / "data" / "raw" / f"vix_daily_{real_cfg['vix_history_start'][:4]}_{real_cfg['crsp_end'][:4]}.parquet"
    crsp_source_name = "local cache"
    vix_source_name = "local cache"
    if crsp_cache.exists() and vix_cache.exists() and not refresh:
        print("[1/8] Loading cached WRDS CRSP and VIX data...")
        crsp = read_parquet_cache(crsp_cache)
        vix_daily = read_parquet_cache(vix_cache)
        if "source_table" in crsp and crsp["source_table"].notna().any():
            table = str(crsp.loc[crsp["source_table"].notna(), "source_table"].iloc[0])
            fmt = str(crsp.loc[crsp["source_format"].notna(), "source_format"].iloc[0]).upper() if "source_format" in crsp and crsp["source_format"].notna().any() else ""
            crsp_source_name = f"{table} ({fmt})".strip()
        if "source_table" in vix_daily and vix_daily["source_table"].notna().any():
            vix_source_name = str(vix_daily.loc[vix_daily["source_table"].notna(), "source_table"].iloc[0])
    else:
        print("[1/8] Connecting to WRDS and resolving current CRSP/Cboe sources...")
        db = connect_wrds()
        try:
            print("[2/8] Resolving authoritative daily Cboe VIX history (WRDS first; official Cboe fallback)...")
            vix_daily, vix_source, _ = cached_vix_daily(db, vix_cache, real_cfg["vix_history_start"], real_cfg["crsp_end"], refresh=refresh)
            if vix_source is not None:
                vix_source_name = f"{vix_source.library}.{vix_source.table} [{vix_source.value_col}]"
            elif "source_table" in vix_daily and vix_daily["source_table"].notna().any():
                vix_source_name = str(vix_daily.loc[vix_daily["source_table"].notna(), "source_table"].iloc[0])
            print("[3/8] Pulling reduced-universe CRSP monthly common stocks...")
            crsp, crsp_source, _ = cached_crsp_monthly(db, crsp_cache, real_cfg["crsp_start"], real_cfg["crsp_end"], min_price=None, top_n_by_market_cap=int(real_cfg["top_n_by_market_cap"]), include_ticker=ticker, universe_formation_date=real_cfg["universe_formation_date"], refresh=refresh)
            if crsp_source is not None:
                crsp_source_name = f"{crsp_source.library}.{crsp_source.table} ({crsp_source.format.upper()})"
        finally:
            db.close()
    if crsp.empty:
        raise RuntimeError("WRDS CRSP extraction returned no observations.")
    if ticker.upper() not in set(crsp["ticker"].dropna().astype(str).str.upper()):
        raise RuntimeError(f"Selected ticker {ticker.upper()} is absent from the extracted CRSP sample.")
    print("[4/8] Building calendar-safe size/momentum predictors and VIX regimes...")
    features = build_features(crsp)
    vix_monthly = vix_daily_to_monthly(vix_daily)
    regimes_all = label_vix_regime(vix_monthly)
    regimes = regimes_all[regimes_all["date"] >= pd.Timestamp(real_cfg["crsp_start"])].copy()
    panel = features.merge(regimes[["date", "vix", "vix_expanding_median", "regime"]], on="date", how="left", validate="many_to_one")
    feature_cols = list(real_cfg["predictors"])
    data_validation_report(panel, feature_cols).to_csv(output_dir / "data_validation.csv", index=False)
    model_panel = panel[panel["prc"] >= float(real_cfg["min_price"])].copy()
    model_panel = model_panel.dropna(subset=[*feature_cols, "next_month_return", "vix", "regime"]).copy()
    start_test = pd.Timestamp(real_cfg["oos_start"])
    initial_train = model_panel[model_panel["date"] < start_test].copy()
    if initial_train["date"].nunique() < int(real_cfg["tuning"]["min_train_months"]):
        raise RuntimeError("Not enough pre-OOS history for the configured forward-chaining tuning stage.")
    print("[5/8] Selecting Elastic Net and XGBoost parameters with historical forward-chaining CV...")
    tuning_tables: list[pd.DataFrame] = []
    selected: dict[str, dict[str, Any]] = {}
    for model_name in ("elastic_net", "xgboost"):
        best, table = select_params_time_cv(initial_train, feature_cols, model_name, real_cfg["tuning"][model_name], n_splits=int(real_cfg["tuning"]["n_folds"]), min_train_months=int(real_cfg["tuning"]["min_train_months"]), random_state=random_state)
        selected[model_name] = best
        tuning_tables.append(table)
    tuning = pd.concat(tuning_tables, ignore_index=True)
    tuning["params"] = tuning["params"].map(str)
    tuning.to_csv(output_dir / "model_selection.csv", index=False)
    selected["elastic_net"].setdefault("max_iter", 10000)
    selected["xgboost"].update({"n_jobs": int(real_cfg.get("xgboost_n_jobs", 4)), "verbosity": 0})
    models = {"elastic_net": ElasticNetModel(selected["elastic_net"], random_state=random_state), "xgboost": XGBoostModel(selected["xgboost"], random_state=random_state)}
    print("[6/8] Generating strict expanding-window cross-sectional OOS predictions...")
    predictions = run_expanding_oos(model_panel, feature_cols=feature_cols, models=models, start_test=start_test, end_test=pd.Timestamp(real_cfg["crsp_end"]), min_train_months=int(real_cfg["min_train_months"]), retrain_every_months=int(real_cfg["retrain_every_months"]))
    validate_oos_predictions(predictions)
    stock = predictions[predictions["ticker"].fillna("").str.upper() == ticker.upper()].copy()
    if stock.empty:
        raise RuntimeError(f"No valid OOS predictions were produced for {ticker.upper()}.")
    print("[7/8] Computing overall/HIGH/LOW metrics and research figures...")
    stock.to_csv(output_dir / "single_stock_predictions_long.csv", index=False)
    _wide_stock_predictions(stock).to_csv(output_dir / "single_stock_predictions.csv", index=False)
    metrics = metrics_table(stock)
    metrics[metrics["regime"] == "OVERALL"].to_csv(output_dir / "single_stock_metrics.csv", index=False)
    metrics[metrics["regime"].isin(["HIGH", "LOW"])].to_csv(output_dir / "regime_metrics.csv", index=False)
    model_comparison(metrics).to_csv(output_dir / "model_comparison.csv", index=False)
    plot_actual_vs_predicted(stock, output_dir / "actual_vs_predicted.png")
    plot_vix_regimes(regimes_all, output_dir / "vix_regimes.png")
    plot_regime_performance(metrics, output_dir / "regime_performance.png")
    metadata: dict[str, Any] = {"mode": "quick" if quick else "production", "selected_ticker": ticker.upper(), "crsp_source": crsp_source_name, "vix_source": vix_source_name, "crsp_rows": int(len(crsp)), "crsp_stocks": int(crsp["permno"].nunique()), "feature_cols": feature_cols, "selected_params": selected, "oos_start": str(start_test.date()), "oos_end": str(pd.to_datetime(stock["date"]).max().date()), "retrain_every_months": int(real_cfg["retrain_every_months"])}
    write_json(metadata, output_dir / "run_metadata.json")
    _write_empirical_summary(output_dir, ticker.upper(), stock, metrics, feature_cols, config, metadata)
    print("[8/8] Preliminary empirical artifacts written successfully.")
    return {"output_dir": output_dir, "predictions": output_dir / "single_stock_predictions.csv", "metrics": output_dir / "single_stock_metrics.csv", "summary": output_dir / "preliminary_results_summary.md"}

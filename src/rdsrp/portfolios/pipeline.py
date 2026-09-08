"""Orchestration for preliminary cross-sectional portfolio evaluation."""
from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from rdsrp.data.cache import read_parquet_cache
from rdsrp.data.vix import cached_vix_daily, vix_daily_to_monthly
from rdsrp.data.wrds import cached_crsp_monthly, connect_wrds
from rdsrp.eval.oos import run_expanding_oos
from rdsrp.features.build import build_features
from rdsrp.models.elastic_net import ElasticNetModel
from rdsrp.models.tuning import select_params_time_cv
from rdsrp.models.xgboost_model import XGBoostModel
from rdsrp.portfolios.evaluation import (
    compute_decile_returns,
    long_short_returns,
    monthly_rank_metrics,
    prepare_portfolio_assignments,
    regime_portfolio_comparison,
    restrict_to_fixed_universe,
    summarize_portfolios,
    summarize_rank_metrics,
    validate_portfolio_outputs,
)
from rdsrp.portfolios.figures import (
    plot_cumulative_long_short,
    plot_decile_return_profile,
    plot_regime_long_short_comparison,
)
from rdsrp.regimes.vix_regime import label_vix_regime
from rdsrp.utils.checks import validate_oos_predictions
from rdsrp.utils.io import ensure_dir
from rdsrp.utils.serialization import write_json


def _load_config(repo_root: Path, *, quick: bool) -> dict[str, Any]:
    path = repo_root / "configs" / ("quick.yaml" if quick else "production.yaml")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Run configuration is invalid: {path}")
    return config


def _source_name(frame: pd.DataFrame, default: str = "local cache") -> str:
    if "source_table" in frame and frame["source_table"].notna().any():
        table = str(frame.loc[frame["source_table"].notna(), "source_table"].iloc[0])
        if "source_format" in frame and frame["source_format"].notna().any():
            fmt = str(frame.loc[frame["source_format"].notna(), "source_format"].iloc[0]).upper()
            return f"{table} ({fmt})"
        return table
    return default


def _load_cached_or_wrds(
    repo_root: Path,
    real_cfg: dict[str, Any],
    *,
    refresh: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    cache_tag = (
        f"{real_cfg['crsp_start'][:4]}_{real_cfg['crsp_end'][:4]}_"
        f"fixedtop{real_cfg['top_n_by_market_cap']}_p{int(real_cfg['min_price'])}"
    )
    crsp_cache = repo_root / "data" / "raw" / f"crsp_monthly_{cache_tag}.parquet"
    vix_cache = (
        repo_root
        / "data"
        / "raw"
        / f"vix_daily_{real_cfg['vix_history_start'][:4]}_{real_cfg['crsp_end'][:4]}.parquet"
    )
    if crsp_cache.exists() and vix_cache.exists() and not refresh:
        crsp = read_parquet_cache(crsp_cache)
        vix_daily = read_parquet_cache(vix_cache)
        return crsp, vix_daily, _source_name(crsp), _source_name(vix_daily)

    db = connect_wrds()
    try:
        vix_daily, vix_source, _ = cached_vix_daily(
            db,
            vix_cache,
            real_cfg["vix_history_start"],
            real_cfg["crsp_end"],
            refresh=refresh,
        )
        crsp, crsp_source, _ = cached_crsp_monthly(
            db,
            crsp_cache,
            real_cfg["crsp_start"],
            real_cfg["crsp_end"],
            min_price=None,
            top_n_by_market_cap=int(real_cfg["top_n_by_market_cap"]),
            include_ticker=None,
            universe_formation_date=real_cfg["universe_formation_date"],
            refresh=refresh,
        )
        crsp_name = (
            f"{crsp_source.library}.{crsp_source.table} ({crsp_source.format.upper()})"
            if crsp_source is not None
            else _source_name(crsp)
        )
        if vix_source is not None:
            vix_name = f"{vix_source.library}.{vix_source.table} [{vix_source.value_col}]"
        else:
            vix_name = _source_name(vix_daily)
        return crsp, vix_daily, crsp_name, vix_name
    finally:
        db.close()


def _prediction_export(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "formation_date",
        "realized_return_date",
        "permno",
        "ticker",
        "model",
        "prediction",
        "actual_next_month_return",
        "regime",
        "me_lag",
        "value_weight_date",
        "train_feature_end_date",
        "train_target_end_date",
    ]
    for col in columns:
        if col not in predictions.columns:
            predictions[col] = pd.NA
    return predictions[columns].sort_values(
        ["formation_date", "model", "permno"]
    ).reset_index(drop=True)


def _validation_lines(
    predictions: pd.DataFrame,
    assignments: pd.DataFrame,
    decile_returns: pd.DataFrame,
    rank_summary: pd.DataFrame,
    skipped: pd.DataFrame,
) -> list[str]:
    lines: list[str] = []
    by_model = predictions.groupby("model").size()
    lines.append("Cross-sectional OOS stock-month observations by model: " + ", ".join(
        f"{model}={int(count)}" for model, count in by_model.items()
    ))
    counts = predictions.groupby(["formation_date", "model"])["permno"].nunique()
    if len(counts):
        lines.append(
            "Monthly stock counts (min/median/max): "
            f"{int(counts.min())}/{float(counts.median()):.1f}/{int(counts.max())}"
        )
    valid = decile_returns[decile_returns["D10_minus_D1"].notna()]
    valid_months = valid.groupby(["model", "weighting"])["formation_date"].nunique()
    lines.append("Valid portfolio months: " + ", ".join(
        f"{model}-{weighting}={int(count)}" for (model, weighting), count in valid_months.items()
    ))
    regime_months = (
        predictions[["formation_date", "model", "regime"]]
        .drop_duplicates()
        .groupby(["model", "regime"])["formation_date"]
        .nunique()
    )
    lines.append("Formation-month regime counts: " + ", ".join(
        f"{model}-{regime}={int(count)}" for (model, regime), count in regime_months.items()
    ))
    lines.append("Every accepted portfolio month contains D1 through D10: yes")
    lines.append("D10-D1 arithmetic checks: passed")
    for weighting in ("EW", "VW"):
        view = decile_returns[decile_returns["weighting"] == weighting]["D10_minus_D1"]
        finite = int(np.isfinite(pd.to_numeric(view, errors="coerce").to_numpy(float)).sum())
        lines.append(f"Finite {weighting} long-short months: {finite}/{len(view)}")
    lines.append("Monthly Spearman summary:")
    for row in rank_summary.itertuples(index=False):
        lines.append(
            f"  {row.model} {row.regime}: n={int(row.n_months)}, "
            f"mean={row.mean_spearman:.6f}, median={row.median_spearman:.6f}, "
            f"sd={row.sd_spearman:.6f}, fraction_positive={row.fraction_positive:.4f}"
        )
    lines.append(f"Skipped month-model groups: {len(skipped)}")
    for row in skipped.itertuples(index=False):
        lines.append(
            f"  {pd.Timestamp(row.formation_date).date()} {row.model}: {row.reason} "
            f"(n={int(row.n_stocks)})"
        )
    return lines


def run_wrds_cross_sectional_evaluation(
    repo_root: Path,
    *,
    quick: bool = True,
    refresh: bool = False,
) -> dict[str, Path]:
    if not quick:
        raise RuntimeError(
            "Production cross-sectional evaluation requires the complete Compustat/CCM predictor panel."
        )
    config = _load_config(repo_root, quick=quick)
    run_cfg = config["run"]
    real_cfg = run_cfg["real"]
    random_state = int(run_cfg.get("random_state", 42))
    output_dir = ensure_dir(repo_root / "results" / "preliminary_cross_section")
    for old in output_dir.iterdir():
        if old.is_file() and old.name != ".gitkeep":
            old.unlink()

    print("[1/6] Loading cached research data and configuration...")
    crsp_raw, vix_daily, crsp_source_name, vix_source_name = _load_cached_or_wrds(
        repo_root, real_cfg, refresh=refresh
    )
    if crsp_raw.empty:
        raise RuntimeError("CRSP extraction/cache contains no observations.")

    crsp = restrict_to_fixed_universe(
        crsp_raw,
        formation_date=real_cfg["universe_formation_date"],
        top_n=int(real_cfg["top_n_by_market_cap"]),
    )
    if crsp["permno"].nunique() != int(real_cfg["top_n_by_market_cap"]):
        raise AssertionError("Fixed cross-sectional universe does not contain the configured top-N PERMNOs.")

    features = build_features(crsp)
    vix_monthly = vix_daily_to_monthly(vix_daily)
    regimes_all = label_vix_regime(vix_monthly)
    regimes = regimes_all[regimes_all["date"] >= pd.Timestamp(real_cfg["crsp_start"])].copy()
    panel = features.merge(
        regimes[["date", "vix", "vix_expanding_median", "regime"]],
        on="date",
        how="left",
        validate="many_to_one",
    )
    feature_cols = list(real_cfg["predictors"])
    model_panel = panel[panel["prc"] >= float(real_cfg["min_price"])].copy()
    model_panel = model_panel.dropna(
        subset=[*feature_cols, "next_month_return", "vix", "regime"]
    ).copy()
    start_test = pd.Timestamp(real_cfg["oos_start"])
    end_test = pd.Timestamp(real_cfg["crsp_end"])
    initial_train = model_panel[model_panel["date"] < start_test].copy()
    if initial_train["date"].nunique() < int(real_cfg["tuning"]["min_train_months"]):
        raise RuntimeError("Not enough pre-OOS history for the configured forward-chaining tuning stage.")

    oos_dates = sorted(
        model_panel.loc[
            (model_panel["date"] >= start_test) & (model_panel["date"] <= end_test), "date"
        ].dropna().unique()
    )
    expected_refits = ceil(len(oos_dates) / int(real_cfg["retrain_every_months"])) if oos_dates else 0
    oos_range = (
        f"{pd.Timestamp(oos_dates[0]).date()} to {pd.Timestamp(oos_dates[-1]).date()}"
        if oos_dates
        else "none"
    )
    print(f"Input rows: {len(model_panel):,}")
    print(f"Unique stocks: {model_panel['permno'].nunique():,}")
    print(f"OOS date range: {oos_range}")
    print(f"Expected refits per model: {expected_refits}")
    print("Models: Elastic Net, XGBoost")

    print("[2/6] Generating or loading cross-sectional OOS predictions...")
    tuning_tables: list[pd.DataFrame] = []
    selected: dict[str, dict[str, Any]] = {}
    for model_name in ("elastic_net", "xgboost"):
        best, table = select_params_time_cv(
            initial_train,
            feature_cols,
            model_name,
            real_cfg["tuning"][model_name],
            n_splits=int(real_cfg["tuning"]["n_folds"]),
            min_train_months=int(real_cfg["tuning"]["min_train_months"]),
            random_state=random_state,
        )
        selected[model_name] = best
        tuning_tables.append(table)
    tuning = pd.concat(tuning_tables, ignore_index=True)
    tuning["params"] = tuning["params"].map(str)
    tuning.to_csv(output_dir / "model_selection.csv", index=False)
    selected["elastic_net"].setdefault("max_iter", 10000)
    selected["xgboost"].update(
        {"n_jobs": int(real_cfg.get("xgboost_n_jobs", 4)), "verbosity": 0}
    )
    models = {
        "elastic_net": ElasticNetModel(selected["elastic_net"], random_state=random_state),
        "xgboost": XGBoostModel(selected["xgboost"], random_state=random_state),
    }
    predictions = run_expanding_oos(
        model_panel,
        feature_cols=feature_cols,
        models=models,
        start_test=start_test,
        end_test=end_test,
        min_train_months=int(real_cfg["min_train_months"]),
        retrain_every_months=int(real_cfg["retrain_every_months"]),
    )
    validate_oos_predictions(predictions)
    cross_predictions = _prediction_export(predictions.copy())
    cross_predictions.to_csv(output_dir / "cross_sectional_predictions.csv", index=False)

    print("[3/6] Assigning monthly predicted-return deciles...")
    assignments, diagnostics = prepare_portfolio_assignments(cross_predictions)
    assignments_export = assignments.rename(
        columns={"actual_next_month_return": "realized_next_month_return"}
    )
    assignments_export.to_csv(output_dir / "portfolio_assignments.csv", index=False)
    diagnostics.skipped_months.to_csv(output_dir / "skipped_months.csv", index=False)

    print("[4/6] Computing equal- and value-weighted portfolio returns...")
    decile_returns = compute_decile_returns(assignments)
    if decile_returns.empty:
        raise RuntimeError("No valid monthly decile portfolios were constructed.")
    decile_returns.to_csv(output_dir / "decile_returns_monthly.csv", index=False)
    long_short = long_short_returns(decile_returns)
    long_short.to_csv(output_dir / "long_short_returns.csv", index=False)
    portfolio_summary = summarize_portfolios(long_short)
    portfolio_summary.to_csv(output_dir / "portfolio_summary.csv", index=False)
    comparison = regime_portfolio_comparison(portfolio_summary)
    comparison.to_csv(output_dir / "regime_portfolio_comparison.csv", index=False)

    print("[5/6] Computing regime and ranking diagnostics...")
    rank_monthly = monthly_rank_metrics(cross_predictions)
    rank_summary = summarize_rank_metrics(rank_monthly)
    rank_monthly.to_csv(output_dir / "cross_sectional_rank_metrics_monthly.csv", index=False)
    rank_summary.to_csv(output_dir / "cross_sectional_rank_metrics_summary.csv", index=False)
    validate_portfolio_outputs(cross_predictions, assignments, decile_returns)

    plot_cumulative_long_short(
        long_short, weighting="EW", path=output_dir / "cumulative_long_short_ew.png"
    )
    plot_cumulative_long_short(
        long_short, weighting="VW", path=output_dir / "cumulative_long_short_vw.png"
    )
    plot_regime_long_short_comparison(
        portfolio_summary,
        weighting="EW",
        path=output_dir / "regime_long_short_comparison.png",
    )
    plot_regime_long_short_comparison(
        portfolio_summary,
        weighting="VW",
        path=output_dir / "regime_long_short_comparison_vw.png",
    )
    plot_decile_return_profile(
        decile_returns, weighting="EW", path=output_dir / "decile_return_profile.png"
    )
    plot_decile_return_profile(
        decile_returns, weighting="VW", path=output_dir / "decile_return_profile_vw.png"
    )

    validation_lines = _validation_lines(
        cross_predictions,
        assignments,
        decile_returns,
        rank_summary,
        diagnostics.skipped_months,
    )
    (output_dir / "validation_report.txt").write_text(
        "\n".join(validation_lines) + "\n", encoding="utf-8"
    )
    metadata: dict[str, Any] = {
        "mode": "quick",
        "stage": "preliminary cross-sectional portfolio evaluation",
        "crsp_source": crsp_source_name,
        "vix_source": vix_source_name,
        "fixed_universe_size": int(real_cfg["top_n_by_market_cap"]),
        "universe_formation_date": real_cfg["universe_formation_date"],
        "min_price": float(real_cfg["min_price"]),
        "feature_cols": feature_cols,
        "value_weight_variable": "me_lag",
        "value_weight_timing": "previous calendar-month market equity, observable before formation month t",
        "selected_params": selected,
        "oos_start": str(start_test.date()),
        "oos_end": str(pd.to_datetime(cross_predictions["formation_date"]).max().date()),
        "retrain_every_months": int(real_cfg["retrain_every_months"]),
        "cross_sectional_observations": int(len(cross_predictions)),
    }
    write_json(metadata, output_dir / "run_metadata.json")

    print("[6/6] Writing preliminary cross-sectional research artifacts...")
    for line in validation_lines:
        print(line)
    print(f"Artifacts written to: {output_dir}")
    return {
        "output_dir": output_dir,
        "predictions": output_dir / "cross_sectional_predictions.csv",
        "assignments": output_dir / "portfolio_assignments.csv",
        "decile_returns": output_dir / "decile_returns_monthly.csv",
        "long_short": output_dir / "long_short_returns.csv",
        "summary": output_dir / "portfolio_summary.csv",
        "comparison": output_dir / "regime_portfolio_comparison.csv",
        "rank_summary": output_dir / "cross_sectional_rank_metrics_summary.csv",
        "validation": output_dir / "validation_report.txt",
    }

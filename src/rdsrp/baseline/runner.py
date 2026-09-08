"""End-to-end orchestration for production baseline estimation."""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rdsrp.models.elastic_net import ElasticNetModel
from rdsrp.models.xgboost_model import XGBoostModel

from .data import (
    BASELINE_PREDICTORS,
    BENCHMARK_DEFINITION,
    BENCHMARK_NAME,
    EXPECTED_PRODUCTION_COUNTS,
    load_production_config,
    prepare_baseline_sample,
    production_count_audit,
    read_production_panel,
)
from .estimation import (
    build_models,
    deterministic_tuning_sample,
    run_expanding_baseline_oos,
    tune_baseline_models,
)
from .reporting import (
    baseline_model_metrics,
    model_comparison_table,
    monthly_rank_correlations,
    prediction_diagnostics,
    summarize_rank_metrics,
    write_figures,
)


@dataclass(frozen=True)
class BaselinePaths:
    output_dir: Path
    checkpoint_dir: Path
    predictions: Path
    metrics: Path
    comparison: Path
    rank_metrics: Path
    diagnostics: Path
    model_selection: Path
    metadata: Path


def output_paths(repo_root: Path) -> BaselinePaths:
    out = repo_root / "results" / "baseline_models"
    return BaselinePaths(
        out,
        out / "checkpoints",
        out / "baseline_oos_predictions.parquet",
        out / "baseline_model_metrics.csv",
        out / "baseline_model_comparison.csv",
        out / "baseline_rank_metrics.csv",
        out / "prediction_diagnostics.csv",
        out / "model_selection.csv",
        out / "run_metadata.json",
    )


def _git_sha(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def configuration_fingerprint(
    *, panel_path: Path, config: dict[str, Any], selected_params: dict[str, dict[str, Any]] | None = None
) -> str:
    stat = panel_path.stat() if panel_path.exists() else None
    source_files = [Path(__file__), Path(__file__).with_name("data.py"), Path(__file__).with_name("estimation.py")]
    implementation_hash = hashlib.sha256(
        b"".join(path.read_bytes() for path in source_files)
    ).hexdigest()
    payload = {
        "panel_path": str(panel_path.resolve()),
        "panel_size": stat.st_size if stat else None,
        "panel_mtime_ns": stat.st_mtime_ns if stat else None,
        "implementation_hash": implementation_hash,
        "config": config,
        "selected_params": selected_params,
        "predictors": BASELINE_PREDICTORS,
        "benchmark": BENCHMARK_NAME,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _clear_outputs(paths: BaselinePaths) -> None:
    for child in paths.checkpoint_dir.glob("*") if paths.checkpoint_dir.exists() else []:
        if child.is_file():
            child.unlink()
    selected = paths.output_dir / "selected_params.json"
    if selected.exists():
        selected.unlink()


def run_production_baseline(repo_root: Path, *, refresh: bool = False) -> dict[str, Path]:
    cfg = load_production_config(repo_root)
    run_cfg, real_cfg = cfg["run"], cfg["run"]["real"]
    random_state = int(run_cfg.get("random_state", 42))
    if str(real_cfg.get("benchmark", BENCHMARK_NAME)) != BENCHMARK_NAME:
        raise ValueError(f"Production benchmark must be {BENCHMARK_NAME!r}.")
    if list(real_cfg.get("predictors", [])) != list(BASELINE_PREDICTORS):
        raise ValueError(f"Production predictors must be exactly {list(BASELINE_PREDICTORS)}.")
    paths = output_paths(repo_root)
    paths.output_dir.mkdir(parents=True, exist_ok=True); paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if refresh:
        _clear_outputs(paths)

    panel_path = repo_root / "data" / "processed" / "modeling_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Completed production panel not found at {panel_path}. Reuse the validated local artifact."
        )

    print("[1/8] Loading and validating the completed production panel...")
    panel = read_production_panel(panel_path)
    memory_mb = float(panel.memory_usage(index=True, deep=True).sum() / 1024**2)
    audit = production_count_audit(
        panel, min_price=float(real_cfg["min_price"]), expected=EXPECTED_PRODUCTION_COUNTS, strict=True
    )
    print(
        f"Loaded {audit['total_stock_months']:,} rows ({memory_mb:,.1f} MiB); "
        f"complete predictors={audit['complete_predictor_rows']:,}; final screened={audit['final_screened_rows']:,}."
    )

    print("[2/8] Preparing the three-predictor baseline modeling sample...")
    sample = prepare_baseline_sample(panel, min_price=float(real_cfg["min_price"])); del panel
    if len(sample) != EXPECTED_PRODUCTION_COUNTS["final_screened_rows"]:
        raise RuntimeError("Prepared baseline sample does not match the frozen screened count.")
    start_test = pd.Timestamp(real_cfg["oos_start"])
    initial_train = sample[(sample["date"] < start_test) & (sample["realized_return_date"] <= start_test)].copy()
    if initial_train["date"].nunique() < int(real_cfg["tuning"]["min_train_months"]):
        raise RuntimeError("Insufficient pre-OOS history for historical parameter selection.")

    print("[3/8] Auditing production VIX regimes and OOS schedule...")
    if sample.groupby("date")["regime"].nunique().max() != 1:
        raise AssertionError("The completed panel contains more than one regime label within a month.")

    base_fingerprint = configuration_fingerprint(panel_path=panel_path, config=cfg)
    selected_path = paths.output_dir / "selected_params.json"
    if selected_path.exists() and not refresh:
        cached = json.loads(selected_path.read_text(encoding="utf-8"))
        if cached.get("base_fingerprint") != base_fingerprint:
            raise RuntimeError("Cached model selection uses a different panel/configuration; use --refresh.")
        print("[4/8] Reusing historically selected baseline hyperparameters from a matching cache...")
        selected = {k: dict(v) for k, v in cached["selected_params"].items()}
    else:
        print("[4/8] Selecting Elastic Net hyperparameters with historical forward-chaining validation...")
        print("[5/8] Selecting XGBoost hyperparameters with historical forward-chaining validation...")
        selected, selection = tune_baseline_models(initial_train, real_cfg, random_state=random_state)
        selection.to_csv(paths.model_selection, index=False)
        selected_path.write_text(
            json.dumps({"base_fingerprint": base_fingerprint, "selected_params": selected}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if not paths.model_selection.exists():
        tune_rows = len(
            deterministic_tuning_sample(
                initial_train,
                max_rows_per_month=int(real_cfg["tuning"].get("max_rows_per_month", 500)),
                random_state=random_state,
            )
        )
        pd.DataFrame(
            [
                {
                    "model": model,
                    "params": json.dumps(params, sort_keys=True),
                    "selected": True,
                    "tuning_start": str(pd.to_datetime(initial_train["date"]).min().date()),
                    "tuning_end": str(pd.to_datetime(initial_train["date"]).max().date()),
                    "tuning_rows": int(tune_rows),
                    "reused_from_cache": True,
                }
                for model, params in selected.items()
            ]
        ).to_csv(paths.model_selection, index=False)

    run_fingerprint = configuration_fingerprint(panel_path=panel_path, config=cfg, selected_params=selected)
    print("[6/8] Generating expanding-window production OOS predictions...")
    predictions, elastic_coef, xgb_importance = run_expanding_baseline_oos(
        sample,
        build_models(selected, random_state=random_state),
        start_test=start_test,
        end_test=pd.Timestamp(real_cfg["crsp_end"]),
        min_train_months=int(real_cfg["min_train_months"]),
        retrain_every_months=int(real_cfg["retrain_every_months"]),
        checkpoint_dir=paths.checkpoint_dir,
        run_fingerprint=run_fingerprint,
        resume=not refresh,
    )
    predictions.to_parquet(paths.predictions, index=False)
    if not elastic_coef.empty:
        elastic_coef.to_csv(paths.output_dir / "elastic_net_coefficients.csv", index=False)
    if not xgb_importance.empty:
        xgb_importance.to_csv(paths.output_dir / "xgboost_feature_importance.csv", index=False)

    print("[7/8] Computing overall, regime, and cross-sectional ranking diagnostics...")
    metrics = baseline_model_metrics(predictions)
    monthly_rank = monthly_rank_correlations(predictions)
    rank_summary = summarize_rank_metrics(monthly_rank)
    diagnostics = prediction_diagnostics(predictions)
    comparison = model_comparison_table(metrics, rank_summary)
    metrics.to_csv(paths.metrics, index=False); comparison.to_csv(paths.comparison, index=False)
    rank_summary.to_csv(paths.rank_metrics, index=False)
    monthly_rank.to_csv(paths.output_dir / "monthly_rank_ic.csv", index=False)
    diagnostics.to_csv(paths.diagnostics, index=False)
    write_figures(predictions, metrics, monthly_rank, paths.output_dir)

    print("[8/8] Writing baseline model research artifacts...")
    regimes = sample[["date", "regime"]].drop_duplicates()
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "production_panel_path": str(panel_path.relative_to(repo_root)),
        "production_panel_version": "validated_1990_2025_complete_panel",
        "git_commit_at_run": _git_sha(repo_root),
        "models": ["Elastic Net", "XGBoost"],
        "predictors": list(BASELINE_PREDICTORS),
        "sample_start": str(pd.to_datetime(sample["date"]).min().date()),
        "sample_end": str(pd.to_datetime(sample["date"]).max().date()),
        "oos_start": str(pd.to_datetime(predictions["date"]).min().date()),
        "oos_end": str(pd.to_datetime(predictions["date"]).max().date()),
        "n_oos_observations": int(len(predictions)),
        "n_unique_stocks_oos": int(predictions["permno"].nunique()),
        "high_months_oos": int(predictions[["date", "regime"]].drop_duplicates()["regime"].eq("HIGH").sum()),
        "low_months_oos": int(predictions[["date", "regime"]].drop_duplicates()["regime"].eq("LOW").sum()),
        "high_months_full_sample": int(regimes["regime"].eq("HIGH").sum()),
        "low_months_full_sample": int(regimes["regime"].eq("LOW").sum()),
        "tuning_scheme": "single historical pre-OOS forward-chaining selection; frozen hyperparameters thereafter",
        "tuning_cross_section_sampling": f"deterministic historical-only cap of {int(real_cfg['tuning'].get('max_rows_per_month', 500))} rows per month",
        "selected_parameters": selected,
        "random_seed": random_state,
        "model_refit_schedule_months": int(real_cfg["retrain_every_months"]),
        "benchmark_name": BENCHMARK_NAME,
        "benchmark_definition": BENCHMARK_DEFINITION,
        "data_source_provenance": {
            "crsp": "crsp.msf_v2 (CIZ)", "compustat": "comp.funda", "ccm": "crsp.ccmxpf_lnkhist", "vix": "cboe_all.cboe"
        },
        "validated_counts": audit,
        "configuration_fingerprint": run_fingerprint,
    }
    paths.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "output_dir": paths.output_dir,
        "predictions": paths.predictions,
        "metrics": paths.metrics,
        "comparison": paths.comparison,
        "rank_metrics": paths.rank_metrics,
        "diagnostics": paths.diagnostics,
        "model_selection": paths.model_selection,
        "metadata": paths.metadata,
    }


def make_validation_panel(
    *, n_stocks: int = 30, n_months: int = 30, random_state: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state); dates = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    rows: list[dict[str, Any]] = []
    for date_index, date in enumerate(dates):
        regime = "HIGH" if date_index % 2 else "LOW"; vix = 28.0 if regime == "HIGH" else 14.0
        for permno in range(1, n_stocks + 1):
            log_me = float(np.log(50 + permno * 3 + date_index)); bm = float(0.3 + (permno % 7) / 10); mom = float(rng.normal(0.08, 0.15))
            target = 0.002 - 0.001 * log_me + 0.004 * bm + 0.02 * mom + float(rng.normal(0.0, 0.03))
            rows.append({"date": date, "permno": permno, "ticker": f"S{permno:03d}", "log_me": log_me, "book_to_market": bm, "mom_12_2": mom, "next_month_return": target, "regime": regime, "vix": vix, "price_eligible": True, "realized_return_date": date + pd.offsets.MonthEnd(1)})
    return pd.DataFrame(rows)


def run_software_validation() -> dict[str, Any]:
    print("Software validation only; these are synthetic values, not empirical results.")
    sample = prepare_baseline_sample(make_validation_panel()); start = sorted(sample["date"].unique())[-4]
    models = {
        "elastic_net": ElasticNetModel({"alpha": 0.001, "l1_ratio": 0.5, "max_iter": 10_000}, random_state=42),
        "xgboost": XGBoostModel({"n_estimators": 30, "max_depth": 2, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "n_jobs": 1, "tree_method": "hist"}, random_state=42),
    }
    predictions, _, _ = run_expanding_baseline_oos(sample, models, start_test=start, end_test=sample["date"].max(), min_train_months=18, retrain_every_months=1)
    metrics = baseline_model_metrics(predictions); monthly = monthly_rank_correlations(predictions); diagnostics = prediction_diagnostics(predictions)
    print(metrics[["model", "regime", "n", "rmse", "oos_r2", "correlation"]].to_string(index=False))
    return {"predictions": predictions, "metrics": metrics, "monthly_rank": monthly, "diagnostics": diagnostics}

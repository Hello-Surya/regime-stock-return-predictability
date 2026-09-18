"""Reduced real-data baseline run and synthetic software validation."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rdsrp.model_spec import FORMATION_DATE, REALIZED_RETURN_DATE
from rdsrp.models.elastic_net import ElasticNetModel
from rdsrp.models.xgboost_model import XGBoostModel

from .data import (
    BASELINE_PREDICTORS,
    BENCHMARK_NAME,
    EXPECTED_PRODUCTION_COUNTS,
    canonical_panel_path,
    load_run_config,
    prepare_baseline_sample,
    production_count_audit,
    read_production_panel,
)
from .estimation import build_models, run_expanding_baseline_oos, tune_baseline_models
from .reporting import (
    baseline_model_metrics,
    monthly_rank_correlations,
    prediction_diagnostics,
    summarize_rank_metrics,
)


def _require_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    run_cfg = cfg["run"]
    real = run_cfg["real"]
    expected = list(BASELINE_PREDICTORS)
    if list(run_cfg["baseline"].get("predictors", [])) != expected:
        raise ValueError(f"run.baseline.predictors must be exactly {expected}")
    if list(real.get("predictors", [])) != expected:
        raise ValueError(f"run.real.predictors must be exactly {expected}")
    if str(real.get("benchmark", BENCHMARK_NAME)) != BENCHMARK_NAME:
        raise ValueError(f"Quick benchmark must be {BENCHMARK_NAME!r}")
    return real


def run_quick_baseline(repo_root: Path, *, refresh: bool = False) -> dict[str, Path]:
    cfg = load_run_config(repo_root, "quick")
    real = _require_contract(cfg)
    panel_path = canonical_panel_path(repo_root)
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Validated canonical baseline panel not found at {panel_path}. "
            "Run scripts/validate_modeling_panel.py --production locally before estimation."
        )
    panel = read_production_panel(panel_path)
    audit = production_count_audit(
        panel,
        min_price=float(real["min_price"]),
        expected=EXPECTED_PRODUCTION_COUNTS,
        strict=True,
    )
    sample = prepare_baseline_sample(panel, min_price=float(real["min_price"]))
    del panel

    universe_date = pd.Timestamp(real["universe_formation_date"]) + pd.offsets.MonthEnd(0)
    cross = sample[sample[FORMATION_DATE] == universe_date]
    if cross.empty:
        raise RuntimeError(f"No complete-case rows at quick universe date {universe_date.date()}")
    top_n = int(real["top_n_by_market_cap"])
    permnos = set(
        cross.sort_values(["log_me", "permno"], ascending=[False, True])
        .drop_duplicates("permno")
        .head(top_n)["permno"]
    )
    sample = sample[sample["permno"].isin(permnos)].copy()

    start = pd.Timestamp(real["oos_start"]) + pd.offsets.MonthEnd(0)
    initial_train = sample[
        (sample[FORMATION_DATE] < start)
        & (sample[REALIZED_RETURN_DATE] <= start)
    ].copy()
    if initial_train[FORMATION_DATE].nunique() < int(real["tuning"]["min_train_months"]):
        raise RuntimeError("Insufficient observable pre-OOS history for quick parameter selection")

    seed = int(cfg["run"].get("random_state", 42))
    selected, selection = tune_baseline_models(initial_train, real, random_state=seed)
    out = repo_root / "results" / "baseline_models" / "quick"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "checkpoints"
    if refresh and checkpoint.exists():
        for child in checkpoint.iterdir():
            if child.is_file():
                child.unlink()
    predictions, _, _ = run_expanding_baseline_oos(
        sample,
        build_models(selected, random_state=seed),
        start_test=start,
        end_test=pd.Timestamp(real["crsp_end"]),
        min_train_months=int(real["min_train_months"]),
        retrain_every_months=int(real["retrain_every_months"]),
        checkpoint_dir=checkpoint,
        run_fingerprint="quick-canonical-v1",
        resume=not refresh,
        max_oos_months=int(real["quick_oos_months"]),
    )
    predictions_path = out / "baseline_oos_predictions.parquet"
    metrics_path = out / "baseline_model_metrics.csv"
    selection_path = out / "model_selection.csv"
    predictions.to_parquet(predictions_path, index=False)
    metrics = baseline_model_metrics(predictions)
    metrics.to_csv(metrics_path, index=False)
    selection.to_csv(selection_path, index=False)
    monthly = monthly_rank_correlations(predictions)
    summarize_rank_metrics(monthly).to_csv(out / "baseline_rank_metrics.csv", index=False)
    prediction_diagnostics(predictions).to_csv(out / "prediction_diagnostics.csv", index=False)
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "quick",
        "interpretation": "Reduced real-data computational validation; not production findings.",
        "production_panel_path": str(panel_path.relative_to(repo_root)),
        "predictors": list(BASELINE_PREDICTORS),
        "model_stage_imputation": "none",
        "leakage_guards": [
            "training formation_date < prediction formation_date",
            "training realized_return_date <= prediction formation_date",
        ],
        "validated_full_panel_counts": audit,
        "selected_parameters": selected,
        "n_oos_observations": int(len(predictions)),
    }
    metadata_path = out / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "output_dir": out,
        "predictions": predictions_path,
        "metrics": metrics_path,
        "model_selection": selection_path,
        "metadata": metadata_path,
    }


def make_validation_panel(
    *, n_stocks: int = 30, n_months: int = 30, random_state: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    for t, date in enumerate(pd.date_range("2010-01-31", periods=n_months, freq="ME")):
        regime = "HIGH" if t % 2 else "LOW"
        for permno in range(1, n_stocks + 1):
            log_me = float(np.log(50 + permno * 3 + t))
            bm = float(0.3 + (permno % 7) / 10)
            mom = float(rng.normal(0.08, 0.15))
            target = 0.002 - 0.001 * log_me + 0.004 * bm + 0.02 * mom + rng.normal(0, 0.03)
            rows.append({
                "date": date,
                FORMATION_DATE: date,
                REALIZED_RETURN_DATE: date + pd.offsets.MonthEnd(1),
                "permno": permno,
                "ticker": f"S{permno:03d}",
                "log_me": log_me,
                "book_to_market": bm,
                "mom_12_2": mom,
                "next_month_return": float(target),
                "regime": regime,
                "vix": 28.0 if regime == "HIGH" else 14.0,
                "price_eligible": True,
            })
    return pd.DataFrame(rows)


def run_software_validation(repo_root: Path | None = None) -> dict[str, Any]:
    print("Software validation only; synthetic values are not empirical results.")
    sample = prepare_baseline_sample(make_validation_panel())
    start = sorted(sample[FORMATION_DATE].unique())[-4]
    models = {
        "elastic_net": ElasticNetModel({"alpha": 0.001, "l1_ratio": 0.5}, random_state=42),
        "xgboost": XGBoostModel({"n_estimators": 20, "max_depth": 2, "n_jobs": 1}, random_state=42),
    }
    predictions, _, _ = run_expanding_baseline_oos(
        sample,
        models,
        start_test=start,
        end_test=sample[FORMATION_DATE].max(),
        min_train_months=18,
        retrain_every_months=1,
    )
    metrics = baseline_model_metrics(predictions)
    monthly = monthly_rank_correlations(predictions)
    diagnostics = prediction_diagnostics(predictions)
    return {
        "predictions": predictions,
        "metrics": metrics,
        "monthly_rank": monthly,
        "diagnostics": diagnostics,
    }

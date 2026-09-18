"""Validate and finalize the canonical baseline modeling panel without running research estimation."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdsrp.data.modeling_panel_validation import (  # noqa: E402
    coverage_by_year_report,
    extreme_observation_reports,
    finalize_baseline_modeling_panel,
    missingness_report,
    model_loader_smoke_test,
    monthly_complete_case_summary,
    monthly_predictor_coverage_report,
    predictor_correlations_report,
    predictor_summary_report,
    sample_flow_report,
    target_summary_report,
    validate_baseline_panel,
    validation_markdown,
)
from rdsrp.data.vix import vix_daily_to_monthly  # noqa: E402
from rdsrp.model_spec import (  # noqa: E402
    BASELINE_FEATURES,
    BASELINE_TARGET,
    FORMATION_DATE,
    REALIZED_RETURN_DATE,
)
from rdsrp.regimes.vix_regime import label_vix_regime  # noqa: E402


def _load_config(mode: str) -> dict[str, Any]:
    path = REPO_ROOT / "configs" / f"{mode}.yaml"
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid configuration: {path}")
    return config


def _assert_model_contract(config: dict[str, Any]) -> None:
    baseline = config.get("run", {}).get("baseline", {})
    expected = {
        "predictors": list(BASELINE_FEATURES),
        "target": BASELINE_TARGET,
        "formation_date": FORMATION_DATE,
        "realized_return_date": REALIZED_RETURN_DATE,
    }
    if baseline != expected:
        raise ValueError(
            "Quick and production configurations must use the centralized baseline input contract. "
            f"Expected {expected}, observed {baseline}."
        )


def _input_path(mode: str) -> Path:
    name = "modeling_panel.parquet" if mode == "production" else "modeling_panel_quick.parquet"
    return REPO_ROOT / "data" / "processed" / name


def _canonical_path(mode: str) -> Path:
    name = (
        "baseline_modeling_panel.parquet"
        if mode == "production"
        else "baseline_modeling_panel_quick.parquet"
    )
    return REPO_ROOT / "data" / "processed" / name


def _validation_dir(mode: str) -> Path:
    base = REPO_ROOT / "results" / "data_validation"
    return base if mode == "production" else base / "quick"


def _reference_regimes(config: dict[str, Any]) -> pd.DataFrame:
    real = config["run"]["real"]
    start_year = str(real["vix_history_start"])[:4]
    end_year = str(real["crsp_end"])[:4]
    path = REPO_ROOT / "data" / "raw" / "vix" / f"daily_{start_year}_{end_year}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "Local VIX history cache is required for an independent historical-regime audit: "
            f"{path}"
        )
    return label_vix_regime(vix_daily_to_monthly(pd.read_parquet(path)))


def _write_outputs(
    panel: pd.DataFrame,
    *,
    mode: str,
    metadata: dict[str, Any],
) -> dict[str, Path]:
    out = _validation_dir(mode)
    out.mkdir(parents=True, exist_ok=True)
    flow = sample_flow_report(panel)
    missing = missingness_report(panel)
    predictors = predictor_summary_report(panel)
    target = target_summary_report(panel)
    yearly = coverage_by_year_report(panel)
    monthly = monthly_predictor_coverage_report(panel)
    correlations = predictor_correlations_report(panel)
    extreme_summary, extreme_audit = extreme_observation_reports(panel)
    monthly_summary = monthly_complete_case_summary(monthly)

    prefix = "baseline_" if mode == "production" else "quick_baseline_"
    paths = {
        "sample_flow": out / f"{prefix}sample_flow.csv",
        "missingness": out / f"{prefix}missingness.csv",
        "predictor_summary": out / f"{prefix}predictor_summary.csv",
        "target_summary": out / f"{prefix}target_summary.csv",
        "yearly": out / f"{prefix}coverage_by_year.csv",
        "monthly": out
        / (
            "monthly_predictor_coverage.csv"
            if mode == "production"
            else "quick_monthly_predictor_coverage.csv"
        ),
        "correlations": out / f"{prefix}predictor_correlations.csv",
        "extreme_summary": out / f"{prefix}extreme_observations_summary.csv",
        "markdown": out / f"{prefix}panel_validation.md",
        "metadata": out / f"{prefix}panel_validation.json",
    }
    audit_path = out / (
        "extreme_observations_audit.csv"
        if mode == "production"
        else "quick_extreme_observations_audit.csv"
    )

    flow.to_csv(paths["sample_flow"], index=False)
    missing.to_csv(paths["missingness"], index=False)
    predictors.to_csv(paths["predictor_summary"], index=False)
    target.to_csv(paths["target_summary"], index=False)
    yearly.to_csv(paths["yearly"], index=False)
    monthly.to_csv(paths["monthly"], index=False)
    correlations.to_csv(paths["correlations"], index=False)
    extreme_summary.to_csv(paths["extreme_summary"], index=False)
    extreme_audit.to_csv(audit_path, index=False)

    final_metadata = {
        **metadata,
        "mode": mode,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_features": list(BASELINE_FEATURES),
        "target": BASELINE_TARGET,
        "monthly_complete_case": monthly_summary,
        "artifact_policy": {
            "aggregate_validation_files": "safe for repository",
            "extreme_observations_audit": "local row-level WRDS-derived file; excluded from Git",
            "canonical_panel": "local row-level WRDS-derived file; excluded from Git",
        },
    }
    paths["metadata"].write_text(
        json.dumps(final_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["markdown"].write_text(
        validation_markdown(final_metadata, missing, monthly_summary),
        encoding="utf-8",
    )
    return paths


def run(mode: str) -> Path:
    config = _load_config(mode)
    _assert_model_contract(config)
    min_price = float(config["run"]["real"]["min_price"])
    source = _input_path(mode)
    canonical = _canonical_path(mode)

    print("[1/8] Loading validated CRSP-Compustat panel...")
    if not source.exists():
        raise FileNotFoundError(
            f"Source construction panel not found at {source}. "
            f"Run build_modeling_panel.py --{mode} first."
        )
    panel = pd.read_parquet(source)

    print("[2/8] Finalizing baseline predictor variables...")
    panel = finalize_baseline_modeling_panel(panel, min_price=min_price)

    print("[3/8] Validating target timing...")
    print("[4/8] Validating accounting and momentum timing...")
    metadata = validate_baseline_panel(
        panel,
        min_price=min_price,
        reference_regimes=_reference_regimes(config),
    )

    print("[5/8] Constructing complete-case baseline sample...")
    if not panel["baseline_complete_case"].fillna(False).any():
        raise RuntimeError("Complete baseline sample is empty.")

    print("[6/8] Running panel integrity checks...")
    metadata["model_loader_smoke_test"] = model_loader_smoke_test(panel)
    regime_months_for_metadata = panel[[FORMATION_DATE, "regime"]].drop_duplicates(FORMATION_DATE)
    metadata["high_observations"] = int(
        panel["regime"].astype("string").str.upper().eq("HIGH").sum()
    )
    metadata["low_observations"] = int(
        panel["regime"].astype("string").str.upper().eq("LOW").sum()
    )
    metadata["high_months"] = int(
        regime_months_for_metadata["regime"].astype("string").str.upper().eq("HIGH").sum()
    )
    metadata["low_months"] = int(
        regime_months_for_metadata["regime"].astype("string").str.upper().eq("LOW").sum()
    )

    print("[7/8] Generating validation artifacts...")
    paths = _write_outputs(panel, mode=mode, metadata=metadata)

    print("[8/8] Writing canonical baseline modeling panel...")
    canonical.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(canonical, index=False)

    monthly = pd.read_csv(paths["monthly"])
    counts = pd.to_numeric(monthly["complete_baseline"], errors="coerce").dropna()
    regime_months = panel[[FORMATION_DATE, "regime"]].drop_duplicates(FORMATION_DATE)
    high_obs = int(panel["regime"].astype("string").str.upper().eq("HIGH").sum())
    low_obs = int(panel["regime"].astype("string").str.upper().eq("LOW").sum())
    high_months = int(regime_months["regime"].astype("string").str.upper().eq("HIGH").sum())
    low_months = int(regime_months["regime"].astype("string").str.upper().eq("LOW").sum())

    print("\nBASELINE_MODELING_PANEL_VALIDATION=PASS")
    print(f"CANONICAL_PANEL={canonical.relative_to(REPO_ROOT)}")
    print(f"FORMATION_RANGE={metadata['formation_start']}..{metadata['formation_end']}")
    print(f"REALIZED_RANGE={metadata['realized_return_start']}..{metadata['realized_return_end']}")
    print(f"PANEL_ROWS={metadata['rows']}")
    print(f"UNIQUE_PERMNOS={metadata['unique_permnos']}")
    print(f"COMPLETE_CASE_ROWS={metadata['complete_case_rows']}")
    print(f"COMPLETE_CASE_PERMNOS={metadata['complete_case_permnos']}")
    print(f"MONTHLY_COMPLETE_MIN={int(counts.min())}")
    print(f"MONTHLY_COMPLETE_MEDIAN={float(counts.median())}")
    print(f"MONTHLY_COMPLETE_MAX={int(counts.max())}")
    print(f"HIGH_OBSERVATIONS={high_obs}")
    print(f"LOW_OBSERVATIONS={low_obs}")
    print(f"HIGH_MONTHS={high_months}")
    print(f"LOW_MONTHS={low_months}")
    print(f"DUPLICATE_SECURITY_MONTHS={metadata['duplicate_security_month_rows']}")
    print("MODEL_LOADER_SMOKE=PASS")
    return canonical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="Validate the reduced real-data panel.")
    mode.add_argument(
        "--production",
        action="store_true",
        help="Validate the complete intended research panel.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run("production" if args.production else "quick")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

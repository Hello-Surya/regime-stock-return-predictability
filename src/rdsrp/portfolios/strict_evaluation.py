"""Strict cross-sectional sorting safeguards for research portfolio evaluation.

The preliminary Elastic Net specification can legitimately collapse to an intercept-only
forecast after time-respecting tuning. In that case deterministic security-ID tie-breaking
would create arbitrary deciles despite the model containing no cross-sectional ranking
information. This module prevents that failure mode while retaining deterministic tie
handling for ordinary partial ties.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rdsrp.portfolios.evaluation import PortfolioDiagnostics, assign_deciles

MIN_PORTFOLIOS = 10
ABS_RANGE_TOL = 1e-12
REL_RANGE_TOL = 1e-10


def _prediction_stats(values: pd.Series) -> tuple[int, int, float, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    n_valid = int(len(numeric))
    n_unique = int(numeric.nunique(dropna=True))
    if not n_valid:
        return 0, 0, float("nan"), float("nan")
    low = float(numeric.min())
    high = float(numeric.max())
    spread = high - low
    scale = max(1.0, abs(low), abs(high))
    tolerance = ABS_RANGE_TOL + REL_RANGE_TOL * scale
    return n_valid, n_unique, spread, tolerance


def prediction_group_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Classify each formation-month/model cross-section as sortable or degenerate."""
    required = {"formation_date", "model", "prediction"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction diagnostics missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for (formation_date, model), block in predictions.groupby(
        ["formation_date", "model"], sort=True, dropna=False
    ):
        n_valid, n_unique, spread, tolerance = _prediction_stats(block["prediction"])
        if n_valid < MIN_PORTFOLIOS:
            status = "fewer_than_10_valid_predictions"
        elif n_unique < MIN_PORTFOLIOS:
            status = "fewer_than_10_distinct_predictions"
        elif not np.isfinite(spread) or spread <= tolerance:
            status = "near_constant_predictions"
        else:
            status = "sortable"
        rows.append(
            {
                "formation_date": pd.Timestamp(formation_date),
                "model": model,
                "n_stocks": int(block["permno"].nunique()) if "permno" in block else int(len(block)),
                "n_valid_predictions": n_valid,
                "n_unique_predictions": n_unique,
                "prediction_range": spread,
                "range_tolerance": tolerance,
                "prediction_status": status,
            }
        )
    return pd.DataFrame(rows)


def prepare_portfolio_assignments_strict(
    predictions: pd.DataFrame,
    *,
    weight_col: str = "me_lag",
) -> tuple[pd.DataFrame, PortfolioDiagnostics]:
    """Assign deciles only when forecasts contain genuine cross-sectional dispersion."""
    required = {
        "formation_date",
        "realized_return_date",
        "permno",
        "model",
        "prediction",
        "actual_next_month_return",
        "regime",
        weight_col,
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Cross-sectional predictions missing columns: {sorted(missing)}")

    frame = predictions.copy()
    frame["formation_date"] = pd.to_datetime(frame["formation_date"])
    frame["realized_return_date"] = pd.to_datetime(frame["realized_return_date"])
    frame = frame.dropna(subset=["prediction", "actual_next_month_return"]).copy()
    diagnostics = prediction_group_diagnostics(frame)
    frame = frame.merge(
        diagnostics[
            [
                "formation_date",
                "model",
                "n_unique_predictions",
                "prediction_range",
                "prediction_status",
            ]
        ],
        on=["formation_date", "model"],
        how="left",
        validate="many_to_one",
    )
    frame["decile"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame["decile_status"] = frame["prediction_status"]

    sortable = frame["prediction_status"].eq("sortable")
    if sortable.any():
        assigned = assign_deciles(
            frame.loc[sortable].drop(columns=["decile", "decile_status"]),
            date_col="formation_date",
        )
        frame.loc[assigned.index, "decile"] = assigned["decile"].astype("Int64")
        frame.loc[assigned.index, "decile_status"] = assigned["decile_status"]

    frame["weighting_variable"] = pd.to_numeric(frame[weight_col], errors="coerce")
    skipped = diagnostics.loc[diagnostics["prediction_status"] != "sortable"].rename(
        columns={"prediction_status": "reason"}
    )
    skipped = skipped[
        [
            "formation_date",
            "model",
            "reason",
            "n_stocks",
            "n_valid_predictions",
            "n_unique_predictions",
            "prediction_range",
        ]
    ].reset_index(drop=True)

    columns = [
        "formation_date",
        "realized_return_date",
        "permno",
        "ticker",
        "model",
        "prediction",
        "decile",
        "decile_status",
        "regime",
        "actual_next_month_return",
        "weighting_variable",
        "n_unique_predictions",
        "prediction_range",
    ]
    for optional in ("value_weight_date", "train_feature_end_date", "train_target_end_date"):
        if optional in frame.columns:
            columns.append(optional)
    if "ticker" not in frame.columns:
        frame["ticker"] = pd.NA
    return frame[columns].copy(), PortfolioDiagnostics(skipped_months=skipped)


def monthly_rank_metrics_strict(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly Spearman IC without calling correlation on constant inputs."""
    required = {
        "formation_date",
        "realized_return_date",
        "model",
        "prediction",
        "actual_next_month_return",
        "regime",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Rank-metric input missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    frame = predictions.copy()
    for (formation_date, model), block in frame.groupby(["formation_date", "model"], sort=True):
        clean = block.dropna(subset=["prediction", "actual_next_month_return"]).copy()
        n_valid, n_unique, spread, _ = _prediction_stats(clean["prediction"])
        actual_unique = int(clean["actual_next_month_return"].nunique(dropna=True))
        if n_valid < 2 or n_unique < 2 or actual_unique < 2:
            spearman = float("nan")
        else:
            spearman = float(
                clean["prediction"].corr(clean["actual_next_month_return"], method="spearman")
            )
        realized_dates = clean["realized_return_date"].dropna().unique()
        regimes = clean["regime"].dropna().unique()
        if len(realized_dates) > 1:
            raise AssertionError("A formation month/model maps to multiple realized-return dates.")
        if len(regimes) > 1:
            raise AssertionError("A formation month/model maps to multiple regimes.")
        rows.append(
            {
                "formation_date": pd.Timestamp(formation_date),
                "realized_return_date": realized_dates[0] if len(realized_dates) else pd.NaT,
                "model": model,
                "regime": regimes[0] if len(regimes) else pd.NA,
                "n_stocks": int(len(clean)),
                "n_unique_predictions": n_unique,
                "prediction_range": spread,
                "spearman": spearman,
            }
        )
    return pd.DataFrame(rows).sort_values(["formation_date", "model"]).reset_index(drop=True)


def run_strict_cross_sectional_evaluation(repo_root, *, quick: bool = True, refresh: bool = False):
    """Run the existing pipeline with strict anti-degeneracy sorting safeguards."""
    from rdsrp.portfolios import pipeline as base_pipeline

    original_prepare = base_pipeline.prepare_portfolio_assignments
    original_rank = base_pipeline.monthly_rank_metrics
    base_pipeline.prepare_portfolio_assignments = prepare_portfolio_assignments_strict
    base_pipeline.monthly_rank_metrics = monthly_rank_metrics_strict
    try:
        return base_pipeline.run_wrds_cross_sectional_evaluation(
            repo_root, quick=quick, refresh=refresh
        )
    finally:
        base_pipeline.prepare_portfolio_assignments = original_prepare
        base_pipeline.monthly_rank_metrics = original_rank

"""Predictive metrics, rank diagnostics, calibration checks, and figures."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_corr(a: pd.Series, b: pd.Series, method: str = "pearson") -> float:
    frame = pd.DataFrame(
        {"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 2 or frame["a"].nunique() < 2 or frame["b"].nunique() < 2:
        return float("nan")
    return float(frame["a"].corr(frame["b"], method=method))


def _metric_row(frame: pd.DataFrame, model: str, regime: str) -> dict[str, Any]:
    a = pd.to_numeric(frame["actual_next_month_return"], errors="coerce")
    p = pd.to_numeric(frame[f"{model}_prediction"], errors="coerce")
    b = pd.to_numeric(frame["benchmark_prediction"], errors="coerce")
    valid = a.notna() & p.notna() & b.notna() & np.isfinite(a) & np.isfinite(p) & np.isfinite(b)
    a, p, b = a[valid].to_numpy(float), p[valid].to_numpy(float), b[valid].to_numpy(float)
    if not len(a):
        raise ValueError(f"No valid rows for {model}/{regime} metrics.")
    errors, benchmark_errors = a - p, a - b
    mse = float(np.mean(errors**2))
    benchmark_mse = float(np.mean(benchmark_errors**2))
    actual_std = float(np.std(a, ddof=1)) if len(a) > 1 else float("nan")
    return {
        "model": "Elastic Net" if model == "elastic_net" else "XGBoost",
        "regime": regime,
        "n": int(len(a)),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(np.mean(np.abs(errors))),
        "oos_r2": float(1.0 - mse / benchmark_mse) if benchmark_mse > 0 else float("nan"),
        "correlation": _safe_corr(pd.Series(a), pd.Series(p)),
        "realized_return_std": actual_std,
        "normalized_rmse": float(math.sqrt(mse) / actual_std) if actual_std > 0 else float("nan"),
        "benchmark_rmse": float(math.sqrt(benchmark_mse)),
    }


def baseline_model_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in ("elastic_net", "xgboost"):
        rows.append(_metric_row(predictions, model, "OVERALL"))
        for regime in ("HIGH", "LOW"):
            subset = predictions[predictions["regime"].astype(str).str.upper() == regime]
            if not subset.empty:
                rows.append(_metric_row(subset, model, regime))
    return pd.DataFrame(rows)


def monthly_rank_correlations(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, group in predictions.groupby("date", sort=True):
        regimes = group["regime"].dropna().astype(str).str.upper().unique()
        if len(regimes) != 1:
            raise AssertionError(f"Regime is not unique within formation month {date}.")
        for model in ("elastic_net", "xgboost"):
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "regime": regimes[0],
                    "model": "Elastic Net" if model == "elastic_net" else "XGBoost",
                    "spearman_ic": _safe_corr(
                        group[f"{model}_prediction"], group["actual_next_month_return"], "spearman"
                    ),
                    "pearson_ic": _safe_corr(
                        group[f"{model}_prediction"], group["actual_next_month_return"], "pearson"
                    ),
                    "n": int(len(group)),
                }
            )
    return pd.DataFrame(rows)


def summarize_rank_metrics(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, model_df in monthly.groupby("model", sort=False):
        for regime in ("OVERALL", "HIGH", "LOW"):
            group = model_df if regime == "OVERALL" else model_df[model_df["regime"] == regime]
            x = pd.to_numeric(group["spearman_ic"], errors="coerce").dropna()
            if x.empty:
                continue
            rows.append(
                {
                    "model": model,
                    "regime": regime,
                    "mean_monthly_spearman_ic": float(x.mean()),
                    "median_spearman_ic": float(x.median()),
                    "std_spearman_ic": float(x.std(ddof=1)) if len(x) > 1 else float("nan"),
                    "fraction_months_ic_gt_zero": float((x > 0).mean()),
                    "n_months": int(len(x)),
                    "mean_monthly_pearson_ic": float(
                        pd.to_numeric(group["pearson_ic"], errors="coerce").mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def prediction_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in ("elastic_net", "xgboost"):
        pred_col = f"{model}_prediction"
        for regime in ("OVERALL", "HIGH", "LOW"):
            group = predictions if regime == "OVERALL" else predictions[predictions["regime"] == regime]
            if group.empty:
                continue
            p = pd.to_numeric(group[pred_col], errors="coerce")
            a = pd.to_numeric(group["actual_next_month_return"], errors="coerce")
            finite = p.notna() & np.isfinite(p)
            clean = p[finite]
            monthly_coverage = group.assign(_valid=finite).groupby("date", sort=True)["_valid"].mean()
            rows.append(
                {
                    "model": "Elastic Net" if model == "elastic_net" else "XGBoost",
                    "regime": regime,
                    "n_rows": int(len(group)),
                    "prediction_coverage": float(finite.mean()),
                    "min_monthly_coverage": float(monthly_coverage.min()),
                    "missing_predictions": int(p.isna().sum()),
                    "infinite_predictions": int((~np.isfinite(p.fillna(0))).sum()),
                    "prediction_mean": float(clean.mean()),
                    "prediction_std": float(clean.std(ddof=1)),
                    "prediction_min": float(clean.min()),
                    "prediction_p01": float(clean.quantile(0.01)),
                    "prediction_p05": float(clean.quantile(0.05)),
                    "prediction_p50": float(clean.quantile(0.50)),
                    "prediction_p95": float(clean.quantile(0.95)),
                    "prediction_p99": float(clean.quantile(0.99)),
                    "prediction_max": float(clean.max()),
                    "actual_mean": float(a.mean()),
                    "actual_std": float(a.std(ddof=1)),
                }
            )
    diagnostics = pd.DataFrame(rows)
    overall = diagnostics[diagnostics["regime"] == "OVERALL"]
    if (overall["prediction_coverage"] < 0.999999).any() or (
        overall["min_monthly_coverage"] < 0.999999
    ).any():
        raise RuntimeError("Prediction coverage is incomplete in at least one model/month.")
    if (overall["prediction_std"].abs() < 1e-12).any():
        raise RuntimeError("A production model generated effectively constant predictions.")
    if (overall["missing_predictions"] > 0).any() or (overall["infinite_predictions"] > 0).any():
        raise RuntimeError("Production predictions contain NaN or infinite values.")
    return diagnostics


def model_comparison_table(metrics: pd.DataFrame, rank_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in ("Elastic Net", "XGBoost"):
        m = metrics[metrics["model"] == model].set_index("regime")
        r = rank_summary[rank_summary["model"] == model].set_index("regime")
        row: dict[str, Any] = {"model": model}
        for regime in ("OVERALL", "HIGH", "LOW"):
            prefix = regime.lower()
            if regime in m.index:
                row[f"{prefix}_rmse"] = float(m.loc[regime, "rmse"])
                row[f"{prefix}_oos_r2"] = float(m.loc[regime, "oos_r2"])
                row[f"{prefix}_normalized_rmse"] = float(m.loc[regime, "normalized_rmse"])
            if regime in r.index:
                row[f"{prefix}_mean_spearman_ic"] = float(r.loc[regime, "mean_monthly_spearman_ic"])
        if "HIGH" in m.index and "LOW" in m.index:
            row["high_minus_low_oos_r2"] = float(m.loc["HIGH", "oos_r2"] - m.loc["LOW", "oos_r2"])
        rows.append(row)
    return pd.DataFrame(rows)


def write_figures(
    predictions: pd.DataFrame, metrics: pd.DataFrame, monthly_rank: pd.DataFrame, output_dir: Path
) -> None:
    rows: list[dict[str, Any]] = []
    for date, group in predictions.groupby("date", sort=True):
        for model in ("elastic_net", "xgboost"):
            err = pd.to_numeric(group["actual_next_month_return"], errors="coerce") - pd.to_numeric(
                group[f"{model}_prediction"], errors="coerce"
            )
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "model": "Elastic Net" if model == "elastic_net" else "XGBoost",
                    "rmse": float(np.sqrt(np.mean(np.square(err)))),
                }
            )
    monthly = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for model, group in monthly.groupby("model"):
        rolling = group.sort_values("date").set_index("date")["rmse"].rolling(12, min_periods=3).mean()
        ax.plot(rolling.index, rolling.values, label=model)
    ax.set(title="Production Out-of-Sample Forecast Error", ylabel="12-month rolling cross-sectional RMSE", xlabel="Formation month")
    ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(output_dir / "baseline_oos_performance.png", dpi=180); plt.close(fig)

    frame = metrics[metrics["regime"].isin(["HIGH", "LOW"])]
    labels = ["Elastic Net", "XGBoost"]; x = np.arange(2); width = 0.36
    high = [float(frame[(frame.model == m) & (frame.regime == "HIGH")]["normalized_rmse"].iloc[0]) for m in labels]
    low = [float(frame[(frame.model == m) & (frame.regime == "LOW")]["normalized_rmse"].iloc[0]) for m in labels]
    fig, ax = plt.subplots(figsize=(8, 5.5)); ax.bar(x - width/2, high, width, label="HIGH VIX"); ax.bar(x + width/2, low, width, label="LOW VIX")
    ax.set_xticks(x, labels); ax.set(ylabel="RMSE / realized-return standard deviation", title="Forecast Error Across Volatility Regimes")
    ax.legend(); ax.grid(axis="y", alpha=0.25); fig.tight_layout()
    fig.savefig(output_dir / "regime_model_performance.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for model, group in monthly_rank.groupby("model"):
        group = group.sort_values("date"); ax.plot(group["date"], group["spearman_ic"], label=model, alpha=0.8)
    ax.axhline(0.0, linewidth=1); ax.set(title="Monthly Cross-Sectional Rank Information Coefficient", ylabel="Spearman rank IC", xlabel="Formation month")
    ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(output_dir / "monthly_rank_ic.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model, label in (("elastic_net", "Elastic Net"), ("xgboost", "XGBoost")):
        values = pd.to_numeric(predictions[f"{model}_prediction"], errors="coerce").dropna()
        lo, hi = values.quantile([0.001, 0.999]); clipped = values.clip(lo, hi)
        ax.hist(clipped, bins=80, density=True, histtype="step", linewidth=1.4, label=label)
    ax.set(title="Distribution of Production OOS Predictions", xlabel="Predicted next-month return (0.1% tails clipped for display only)", ylabel="Density")
    ax.legend(); ax.grid(alpha=0.2); fig.tight_layout()
    fig.savefig(output_dir / "prediction_distribution.png", dpi=180); plt.close(fig)

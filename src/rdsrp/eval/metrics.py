"""Prediction metrics for overall and regime-specific OOS evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def prediction_metrics(actual: pd.Series, predicted: pd.Series, benchmark: pd.Series | None = None) -> dict[str, float]:
    a = pd.to_numeric(actual, errors="coerce")
    p = pd.to_numeric(predicted, errors="coerce")
    valid = a.notna() & p.notna()
    if benchmark is not None:
        b = pd.to_numeric(benchmark, errors="coerce")
        valid &= b.notna()
    else:
        b = pd.Series(0.0, index=a.index)
    a = a[valid].to_numpy(dtype=float); p = p[valid].to_numpy(dtype=float); b = b[valid].to_numpy(dtype=float)
    if len(a) == 0:
        return {"n_obs": 0.0, "mse": np.nan, "rmse": np.nan, "mae": np.nan, "r2": np.nan, "oos_r2": np.nan, "oos_r2_vs_zero": np.nan, "correlation": np.nan}
    mse = mean_squared_error(a, p)
    benchmark_sse = float(np.sum((a - b) ** 2)); zero_sse = float(np.sum(a**2))
    corr = float(np.corrcoef(a, p)[0, 1]) if len(a) > 1 and np.std(a) > 0 and np.std(p) > 0 else np.nan
    return {"n_obs": float(len(a)), "mse": float(mse), "rmse": float(np.sqrt(mse)), "mae": float(mean_absolute_error(a, p)), "r2": float(r2_score(a, p)) if len(a) > 1 else np.nan, "oos_r2": float(1.0 - np.sum((a - p) ** 2) / benchmark_sse) if benchmark_sse > 0 else np.nan, "oos_r2_vs_zero": float(1.0 - np.sum((a - p) ** 2) / zero_sse) if zero_sse > 0 else np.nan, "correlation": corr}


def metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, model_df in predictions.groupby("model"):
        benchmark = model_df["benchmark_prediction"] if "benchmark_prediction" in model_df else None
        rows.append({"model": model, "regime": "OVERALL", **prediction_metrics(model_df["actual"], model_df["prediction"], benchmark)})
        for regime, grp in model_df.groupby("regime"):
            regime_benchmark = grp["benchmark_prediction"] if "benchmark_prediction" in grp else None
            rows.append({"model": model, "regime": regime, **prediction_metrics(grp["actual"], grp["prediction"], regime_benchmark)})
    return pd.DataFrame(rows)

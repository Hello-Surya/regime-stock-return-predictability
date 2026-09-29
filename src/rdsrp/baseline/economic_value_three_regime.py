"""Gross economic-value evaluation across binary and three-state VIX regimes.

This module consumes frozen baseline out-of-sample predictions. It never refits,
retunes, or alters the prediction models. The original expanding-median HIGH/LOW
regime remains the baseline; LOW/MIDDLE/HIGH expanding-tercile states are an
additional conditional-evaluation specification.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_COLUMNS: dict[str, str] = {
    "Elastic Net": "elastic_net_prediction",
    "XGBoost": "xgboost_prediction",
}
REGIME_3STATE_ORDER = ("LOW", "MIDDLE", "HIGH")
BINARY_ORDER = ("LOW", "HIGH")
QUANTILE_METHOD = "linear"


def _formation_column(frame: pd.DataFrame) -> str:
    if "formation_date" in frame:
        return "formation_date"
    if "date" in frame:
        return "date"
    raise ValueError("Expected formation_date or date in prediction/VIX data.")


def _finite_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(np.isfinite(numeric))


def classify_three_state_vix(vix: float, q33: float, q67: float) -> str:
    """Classify one formation-month VIX observation using documented boundaries."""
    values = np.asarray([vix, q33, q67], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("vix, q33, and q67 must be finite.")
    if q33 > q67:
        raise AssertionError("q33 cannot exceed q67.")
    if vix <= q33:
        return "LOW"
    if vix <= q67:
        return "MIDDLE"
    return "HIGH"


def build_expanding_vix_regimes(
    vix_monthly: pd.DataFrame,
    *,
    date_col: str = "date",
    vix_col: str = "vix",
    min_periods: int = 1,
) -> pd.DataFrame:
    """Construct expanding-median and expanding-tercile VIX classifications.

    Thresholds include the formation-month observation, matching the repository's
    existing expanding-median rule. q33 and q67 use ``numpy.quantile`` with the
    deterministic ``method='linear'`` convention. No future VIX observations can
    affect an earlier threshold.
    """
    if min_periods < 1:
        raise ValueError("min_periods must be at least one.")
    if vix_monthly.empty:
        return pd.DataFrame(
            columns=[
                date_col,
                vix_col,
                "vix_expanding_median",
                "vix_expanding_q33",
                "vix_expanding_q67",
                "regime_binary",
                "regime_3state",
            ]
        )
    if date_col not in vix_monthly or vix_col not in vix_monthly:
        raise ValueError(f"VIX data must contain '{date_col}' and '{vix_col}'.")

    out = vix_monthly[[date_col, vix_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.to_period("M").dt.to_timestamp("M")
    out[vix_col] = _finite_numeric(out[vix_col])
    out = out.dropna(subset=[date_col, vix_col]).sort_values(date_col, kind="mergesort")
    if out.empty:
        raise ValueError("No finite monthly VIX observations are available.")
    duplicates = out.groupby(date_col)[vix_col].nunique(dropna=True)
    if (duplicates > 1).any():
        bad = duplicates[duplicates > 1].index[0]
        raise AssertionError(f"More than one VIX value exists for month {pd.Timestamp(bad).date()}.")
    out = out.drop_duplicates(date_col, keep="last").reset_index(drop=True)

    values = out[vix_col].to_numpy(float)
    medians = np.full(len(out), np.nan, dtype=float)
    q33 = np.full(len(out), np.nan, dtype=float)
    q67 = np.full(len(out), np.nan, dtype=float)
    for idx in range(len(values)):
        history = values[: idx + 1]
        history = history[np.isfinite(history)]
        if len(history) < min_periods:
            continue
        medians[idx] = float(np.quantile(history, 0.5, method=QUANTILE_METHOD))
        q33[idx] = float(np.quantile(history, 1.0 / 3.0, method=QUANTILE_METHOD))
        q67[idx] = float(np.quantile(history, 2.0 / 3.0, method=QUANTILE_METHOD))

    out["vix_expanding_median"] = medians
    out["vix_expanding_q33"] = q33
    out["vix_expanding_q67"] = q67
    valid = out[["vix_expanding_q33", "vix_expanding_q67"]].notna().all(axis=1)
    if (out.loc[valid, "vix_expanding_q33"] > out.loc[valid, "vix_expanding_q67"]).any():
        raise AssertionError("Expanding q33 exceeds q67 for at least one month.")

    out["regime_binary"] = pd.Series(pd.NA, index=out.index, dtype="string")
    median_valid = out["vix_expanding_median"].notna()
    out.loc[median_valid, "regime_binary"] = np.where(
        out.loc[median_valid, vix_col] > out.loc[median_valid, "vix_expanding_median"],
        "HIGH",
        "LOW",
    )
    out["regime_3state"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out.loc[valid, "regime_3state"] = [
        classify_three_state_vix(vix, low, high)
        for vix, low, high in zip(
            out.loc[valid, vix_col],
            out.loc[valid, "vix_expanding_q33"],
            out.loc[valid, "vix_expanding_q67"],
            strict=True,
        )
    ]
    return out


def monthly_vix_history_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Extract one audited formation-month VIX observation from a panel."""
    date_col = _formation_column(panel)
    if "vix" not in panel:
        raise ValueError("Canonical panel must contain vix.")
    frame = panel[[date_col, "vix"]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col]).dt.to_period("M").dt.to_timestamp("M")
    frame["vix"] = _finite_numeric(frame["vix"])
    frame = frame.dropna(subset=[date_col, "vix"])
    counts = frame.groupby(date_col)["vix"].nunique(dropna=True)
    if (counts != 1).any():
        bad = counts[counts != 1].index[0]
        raise AssertionError(f"VIX is not unique within formation month {pd.Timestamp(bad).date()}.")
    out = frame.groupby(date_col, as_index=False, sort=True)["vix"].first()
    return out.rename(columns={date_col: "date"})


def attach_volatility_regimes(
    predictions: pd.DataFrame,
    vix_history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach three-state thresholds while proving the binary labels are unchanged."""
    required = {
        "permno",
        "regime",
        "vix",
        "actual_next_month_return",
        "me_lag",
        "me_lag_date",
        *MODEL_COLUMNS.values(),
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Prediction artifact is missing required columns: {missing}")
    formation_col = _formation_column(predictions)
    frame = predictions.copy()
    frame[formation_col] = pd.to_datetime(frame[formation_col]).dt.to_period("M").dt.to_timestamp("M")
    if "formation_date" not in frame:
        frame["formation_date"] = frame[formation_col]
    else:
        frame["formation_date"] = pd.to_datetime(frame["formation_date"]).dt.to_period("M").dt.to_timestamp("M")
    if "date" in frame:
        canonical_date = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp("M")
        if not canonical_date.eq(frame["formation_date"]).all():
            raise AssertionError("date and formation_date disagree in saved predictions.")
    if frame.duplicated(["permno", "formation_date"]).any():
        raise AssertionError("Prediction artifact contains duplicate permno/formation_date rows.")

    frame["me_lag_date"] = pd.to_datetime(frame["me_lag_date"]).dt.to_period("M").dt.to_timestamp("M")
    expected_weight_date = frame["formation_date"] + pd.offsets.MonthEnd(-1)
    if not frame["me_lag_date"].eq(expected_weight_date).all():
        raise AssertionError("me_lag_date must be the previous calendar month-end.")
    if "realized_return_date" in frame:
        frame["realized_return_date"] = pd.to_datetime(frame["realized_return_date"]).dt.to_period("M").dt.to_timestamp("M")
        expected_realized = frame["formation_date"] + pd.offsets.MonthEnd(1)
        if not frame["realized_return_date"].eq(expected_realized).all():
            raise AssertionError("realized_return_date must be calendar month t+1.")
    else:
        frame["realized_return_date"] = frame["formation_date"] + pd.offsets.MonthEnd(1)

    regimes = build_expanding_vix_regimes(vix_history, date_col="date", vix_col="vix")
    threshold_cols = [
        "date",
        "vix",
        "vix_expanding_median",
        "vix_expanding_q33",
        "vix_expanding_q67",
        "regime_binary",
        "regime_3state",
    ]
    renamed = regimes[threshold_cols].rename(columns={"date": "formation_date", "vix": "vix_history"})
    merged = frame.merge(renamed, on="formation_date", how="left", validate="many_to_one")
    if merged["regime_3state"].isna().any():
        missing_months = merged.loc[merged["regime_3state"].isna(), "formation_date"].drop_duplicates()
        raise RuntimeError(f"Missing three-state VIX classification for {len(missing_months)} OOS months.")
    prediction_vix = pd.to_numeric(merged["vix"], errors="coerce").to_numpy(float)
    history_vix = pd.to_numeric(merged["vix_history"], errors="coerce").to_numpy(float)
    vix_close = np.isclose(
        prediction_vix,
        history_vix,
        rtol=1e-6,
        atol=1e-6,
        equal_nan=False,
    )
    if not vix_close.all():
        first_bad = int(np.flatnonzero(~vix_close)[0])
        bad_date = pd.Timestamp(merged.iloc[first_bad]["formation_date"]).date()
        max_abs_diff = float(np.nanmax(np.abs(prediction_vix - history_vix)))
        raise AssertionError(
            "Prediction VIX values disagree with the historical VIX series beyond "
            f"float-storage tolerance; first mismatch={bad_date}, "
            f"prediction={prediction_vix[first_bad]:.10g}, "
            f"history={history_vix[first_bad]:.10g}, max_abs_diff={max_abs_diff:.10g}."
        )
    old_binary = merged["regime"].astype(str).str.upper()
    new_binary = merged["regime_binary"].astype(str).str.upper()
    if not old_binary.eq(new_binary).all():
        bad = merged.loc[~old_binary.eq(new_binary), "formation_date"].iloc[0]
        raise AssertionError(f"Three-state extension would alter the existing binary regime at {bad}.")
    merged = merged.drop(columns=["vix_history"])
    return merged, regimes


def assign_deterministic_deciles(
    group: pd.DataFrame,
    prediction_col: str,
    *,
    n_portfolios: int = 10,
) -> pd.Series:
    """Assign approximately equal deciles after prediction/PERMNO stable ordering."""
    if n_portfolios < 2:
        raise ValueError("n_portfolios must be at least two.")
    result = pd.Series(pd.NA, index=group.index, dtype="Int64")
    pred = _finite_numeric(group[prediction_col])
    permno = pd.to_numeric(group["permno"], errors="coerce")
    valid = pred.notna() & permno.notna()
    if int(valid.sum()) < n_portfolios:
        raise RuntimeError(
            f"At least {n_portfolios} valid stocks are required to form {n_portfolios} portfolios."
        )
    ordered = group.loc[valid].assign(_prediction=pred.loc[valid], _permno=permno.loc[valid])
    ordered = ordered.sort_values(["_prediction", "_permno"], kind="mergesort")
    ordinal = np.arange(len(ordered), dtype=int)
    deciles = np.floor(ordinal * n_portfolios / len(ordered)).astype(int) + 1
    result.loc[ordered.index] = deciles
    return result


def build_portfolio_assignments(predictions: pd.DataFrame) -> pd.DataFrame:
    """Create row-level monthly prediction-decile assignments for both models."""
    rows: list[pd.DataFrame] = []
    for model, prediction_col in MODEL_COLUMNS.items():
        for _, raw in predictions.groupby("formation_date", sort=True):
            group = raw.copy()
            realized = _finite_numeric(group["actual_next_month_return"])
            prediction = _finite_numeric(group[prediction_col])
            valid = realized.notna() & prediction.notna()
            group = group.loc[valid].copy()
            group["decile"] = assign_deterministic_deciles(group, prediction_col)
            group = group.dropna(subset=["decile"]).copy()
            out = pd.DataFrame(
                {
                    "formation_date": group["formation_date"],
                    "realized_return_date": group["realized_return_date"],
                    "permno": group["permno"],
                    "ticker": group["ticker"] if "ticker" in group else pd.NA,
                    "model": model,
                    "prediction": prediction.loc[group.index].astype(float),
                    "decile": group["decile"].astype(int),
                    "realized_next_month_return": realized.loc[group.index].astype(float),
                    "weighting_variable": pd.to_numeric(group["me_lag"], errors="coerce"),
                    "vix": pd.to_numeric(group["vix"], errors="coerce"),
                    "vix_expanding_median": group["vix_expanding_median"],
                    "vix_expanding_q33": group["vix_expanding_q33"],
                    "vix_expanding_q67": group["vix_expanding_q67"],
                    "regime_binary": group["regime_binary"].astype(str),
                    "regime_3state": group["regime_3state"].astype(str),
                }
            )
            rows.append(out)
    if not rows:
        raise RuntimeError("No valid portfolio assignments could be formed.")
    assignments = pd.concat(rows, ignore_index=True)
    if assignments.duplicated(["formation_date", "permno", "model"]).any():
        raise AssertionError("Portfolio assignments are not unique by month/security/model.")
    return assignments.sort_values(["formation_date", "model", "decile", "permno"], kind="mergesort").reset_index(drop=True)


def _decile_return(group: pd.DataFrame, decile: int, weighting: str) -> tuple[float, int]:
    side = group[group["decile"] == decile].copy()
    returns = _finite_numeric(side["realized_next_month_return"])
    if weighting == "EW":
        valid = returns.notna()
        if not valid.any():
            return float("nan"), 0
        return float(returns.loc[valid].mean()), int(valid.sum())
    if weighting != "VW":
        raise ValueError("weighting must be 'EW' or 'VW'.")
    weights = _finite_numeric(side["weighting_variable"])
    valid = returns.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return float("nan"), 0
    w = weights.loc[valid].to_numpy(float)
    w /= w.sum()
    return float(np.dot(w, returns.loc[valid].to_numpy(float))), int(valid.sum())


def monthly_decile_returns(assignments: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly D1-D10 gross realized returns using EW and VW weights."""
    rows: list[dict[str, Any]] = []
    for (date, model), group in assignments.groupby(["formation_date", "model"], sort=True):
        binary = group["regime_binary"].dropna().unique()
        three = group["regime_3state"].dropna().unique()
        realized_dates = pd.to_datetime(group["realized_return_date"]).dropna().unique()
        if len(binary) != 1 or len(three) != 1 or len(realized_dates) != 1:
            raise AssertionError("Portfolio month has inconsistent regime or realization timing.")
        for weighting in ("EW", "VW"):
            row: dict[str, Any] = {
                "formation_date": pd.Timestamp(date),
                "realized_return_date": pd.Timestamp(realized_dates[0]),
                "model": model,
                "regime_binary": str(binary[0]),
                "regime_3state": str(three[0]),
                "weighting": weighting,
            }
            counts: list[int] = []
            for decile in range(1, 11):
                value, count = _decile_return(group, decile, weighting)
                if not np.isfinite(value):
                    raise RuntimeError(f"Missing D{decile} return for {model}/{weighting}/{date}.")
                row[f"D{decile}"] = value
                row[f"n_D{decile}"] = count
                counts.append(count)
            row["D10_minus_D1"] = float(row["D10"] - row["D1"])
            row["n_stocks"] = int(sum(counts))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["formation_date", "model", "weighting"]).reset_index(drop=True)


def _summary_row(group: pd.DataFrame, *, model: str, weighting: str, regime: str) -> dict[str, Any]:
    spread = _finite_numeric(group["D10_minus_D1"]).dropna()
    if spread.empty:
        raise ValueError(f"No finite long-short returns for {model}/{weighting}/{regime}.")
    std = float(spread.std(ddof=1)) if len(spread) > 1 else float("nan")
    mean = float(spread.mean())
    return {
        "model": model,
        "regime": regime,
        "weighting": weighting,
        "n_months": int(len(spread)),
        "mean_D1": float(_finite_numeric(group["D1"]).mean()),
        "mean_D10": float(_finite_numeric(group["D10"]).mean()),
        "mean_D10_minus_D1": mean,
        "sd_D10_minus_D1": std,
        "annualized_D10_minus_D1": 12.0 * mean,
        "annualized_volatility": math.sqrt(12.0) * std if np.isfinite(std) else float("nan"),
        "descriptive_sharpe": math.sqrt(12.0) * mean / std if std > 0 else float("nan"),
        "fraction_positive": float((spread > 0).mean()),
        "min_monthly_spread": float(spread.min()),
        "max_monthly_spread": float(spread.max()),
    }


def summarize_portfolios(monthly: pd.DataFrame, regime_definition: str) -> pd.DataFrame:
    """Summarize overall, binary, or three-state gross portfolio performance."""
    if regime_definition not in {"overall", "binary", "three_state"}:
        raise ValueError("regime_definition must be overall, binary, or three_state.")
    rows: list[dict[str, Any]] = []
    regimes = ("OVERALL",) if regime_definition == "overall" else (
        BINARY_ORDER if regime_definition == "binary" else REGIME_3STATE_ORDER
    )
    regime_col = None if regime_definition == "overall" else (
        "regime_binary" if regime_definition == "binary" else "regime_3state"
    )
    for model in MODEL_COLUMNS:
        for weighting in ("EW", "VW"):
            base = monthly[(monthly["model"] == model) & (monthly["weighting"] == weighting)]
            for regime in regimes:
                group = base if regime_col is None else base[base[regime_col] == regime]
                if group.empty:
                    rows.append(
                        {
                            "model": model,
                            "regime": regime,
                            "weighting": weighting,
                            "n_months": 0,
                            "mean_D1": float("nan"),
                            "mean_D10": float("nan"),
                            "mean_D10_minus_D1": float("nan"),
                            "sd_D10_minus_D1": float("nan"),
                            "annualized_D10_minus_D1": float("nan"),
                            "annualized_volatility": float("nan"),
                            "descriptive_sharpe": float("nan"),
                            "fraction_positive": float("nan"),
                            "min_monthly_spread": float("nan"),
                            "max_monthly_spread": float("nan"),
                        }
                    )
                    continue
                rows.append(_summary_row(group, model=model, weighting=weighting, regime=regime))
    return pd.DataFrame(rows)


def monthly_rank_ic(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, group in predictions.groupby("formation_date", sort=True):
        binary = group["regime_binary"].dropna().unique()
        three = group["regime_3state"].dropna().unique()
        if len(binary) != 1 or len(three) != 1:
            raise AssertionError(f"Regime is not unique within formation month {date}.")
        actual = _finite_numeric(group["actual_next_month_return"])
        for model, pred_col in MODEL_COLUMNS.items():
            pred = _finite_numeric(group[pred_col])
            valid = pred.notna() & actual.notna()
            if int(valid.sum()) < 2:
                spearman = float("nan")
            else:
                spearman = float(pred.loc[valid].corr(actual.loc[valid], method="spearman"))
            rows.append(
                {
                    "formation_date": pd.Timestamp(date),
                    "model": model,
                    "regime_binary": str(binary[0]),
                    "regime_3state": str(three[0]),
                    "spearman_ic": spearman,
                    "n_stocks": int(valid.sum()),
                }
            )
    return pd.DataFrame(rows)


def summarize_rank_ic(monthly_rank: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    definitions = [
        ("overall", None, ("OVERALL",)),
        ("binary", "regime_binary", BINARY_ORDER),
        ("three_state", "regime_3state", REGIME_3STATE_ORDER),
    ]
    for model in MODEL_COLUMNS:
        base = monthly_rank[monthly_rank["model"] == model]
        for definition, column, regimes in definitions:
            for regime in regimes:
                group = base if column is None else base[base[column] == regime]
                values = _finite_numeric(group["spearman_ic"]).dropna()
                rows.append(
                    {
                        "model": model,
                        "regime_definition": definition,
                        "regime": regime,
                        "n_months": int(len(values)),
                        "mean_spearman": float(values.mean()) if not values.empty else float("nan"),
                        "median_spearman": float(values.median()) if not values.empty else float("nan"),
                        "sd_spearman": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                        "fraction_positive": float((values > 0).mean()) if not values.empty else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def decile_return_profiles(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODEL_COLUMNS:
        for weighting in ("EW", "VW"):
            base = monthly[(monthly["model"] == model) & (monthly["weighting"] == weighting)]
            for definition, column, regimes in (
                ("overall", None, ("OVERALL",)),
                ("three_state", "regime_3state", REGIME_3STATE_ORDER),
            ):
                for regime in regimes:
                    group = base if column is None else base[base[column] == regime]
                    if group.empty:
                        continue
                    for decile in range(1, 11):
                        values = _finite_numeric(group[f"D{decile}"]).dropna()
                        rows.append(
                            {
                                "model": model,
                                "weighting": weighting,
                                "regime_definition": definition,
                                "regime": regime,
                                "decile": decile,
                                "mean_realized_return": float(values.mean()),
                                "n_months": int(len(values)),
                            }
                        )
    return pd.DataFrame(rows)


def monotonicity_diagnostics(profiles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["model", "weighting", "regime_definition", "regime"]
    for key, group in profiles.groupby(keys, sort=False):
        ordered = group.sort_values("decile")
        returns = ordered["mean_realized_return"].to_numpy(float)
        deciles = ordered["decile"].to_numpy(float)
        if len(returns) != 10:
            continue
        spearman = float(pd.Series(deciles).corr(pd.Series(returns), method="spearman"))
        adjacent_increases = int(np.sum(np.diff(returns) > 0))
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "decile_return_spearman": spearman,
                "adjacent_increases": adjacent_increases,
                "adjacent_comparisons": 9,
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def regime_transition_counts(regimes: pd.DataFrame) -> pd.DataFrame:
    frame = regimes.dropna(subset=["regime_3state"]).sort_values("date").copy()
    frame["from_regime"] = frame["regime_3state"].shift(1)
    frame = frame.dropna(subset=["from_regime"])
    counts = (
        frame.groupby(["from_regime", "regime_3state"], observed=False)
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={"regime_3state": "to_regime"})
    )
    grid = pd.MultiIndex.from_product(
        [REGIME_3STATE_ORDER, REGIME_3STATE_ORDER], names=["from_regime", "to_regime"]
    ).to_frame(index=False)
    out = grid.merge(counts, on=["from_regime", "to_regime"], how="left")
    out["count"] = out["count"].fillna(0).astype(int)
    return out


def regime_validation_table(regimes: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    oos_months = predictions[["formation_date", "regime_3state"]].drop_duplicates()
    stock_counts = predictions.groupby("regime_3state").size()
    month_counts = oos_months.groupby("regime_3state").size()
    valid_thresholds = regimes.dropna(subset=["vix_expanding_q33", "vix_expanding_q67"])
    rows: list[dict[str, Any]] = []
    for regime in REGIME_3STATE_ORDER:
        rows.append(
            {
                "regime": regime,
                "oos_month_count": int(month_counts.get(regime, 0)),
                "oos_stock_month_count": int(stock_counts.get(regime, 0)),
                "earliest_valid_q33": valid_thresholds["date"].min(),
                "earliest_valid_q67": valid_thresholds["date"].min(),
                "q33_min": float(valid_thresholds["vix_expanding_q33"].min()),
                "q33_median": float(valid_thresholds["vix_expanding_q33"].median()),
                "q33_max": float(valid_thresholds["vix_expanding_q33"].max()),
                "q67_min": float(valid_thresholds["vix_expanding_q67"].min()),
                "q67_median": float(valid_thresholds["vix_expanding_q67"].median()),
                "q67_max": float(valid_thresholds["vix_expanding_q67"].max()),
                "q33_le_q67_all_valid": bool(
                    (valid_thresholds["vix_expanding_q33"] <= valid_thresholds["vix_expanding_q67"]).all()
                ),
                "quantile_method": QUANTILE_METHOD,
                "threshold_includes_current_month": True,
                "minimum_history_observations": 1,
            }
        )
    return pd.DataFrame(rows)


def central_three_regime_summary(
    three_summary: pd.DataFrame,
    rank_summary: pd.DataFrame,
) -> pd.DataFrame:
    rank = rank_summary[rank_summary["regime_definition"] == "three_state"][
        ["model", "regime", "mean_spearman"]
    ].rename(columns={"mean_spearman": "mean_rank_ic"})
    return three_summary.merge(rank, on=["model", "regime"], how="left")[
        [
            "model",
            "weighting",
            "regime",
            "n_months",
            "mean_D1",
            "mean_D10",
            "mean_D10_minus_D1",
            "annualized_D10_minus_D1",
            "annualized_volatility",
            "descriptive_sharpe",
            "fraction_positive",
            "mean_rank_ic",
        ]
    ]


def _plot_three_regime_bars(three_summary: pd.DataFrame, output_path: Path, weighting: str) -> None:
    subset = three_summary[three_summary["weighting"] == weighting]
    x = np.arange(len(REGIME_3STATE_ORDER), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for offset, model in zip((-width / 2, width / 2), MODEL_COLUMNS, strict=True):
        values = []
        for regime in REGIME_3STATE_ORDER:
            row = subset[(subset["model"] == model) & (subset["regime"] == regime)]
            values.append(float(row["mean_D10_minus_D1"].iloc[0]))
        ax.bar(x + offset, values, width, label=model)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x, REGIME_3STATE_ORDER)
    ax.set_ylabel("Mean monthly D10-D1 return")
    ax.set_title(f"Three-state VIX economic value ({weighting})")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_figures(
    monthly: pd.DataFrame,
    profiles: pd.DataFrame,
    regimes: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    three_summary = summarize_portfolios(monthly, "three_state")
    _plot_three_regime_bars(three_summary, output_dir / "three_regime_long_short_ew.png", "EW")
    _plot_three_regime_bars(three_summary, output_dir / "three_regime_long_short_vw.png", "VW")

    overall = profiles[profiles["regime_definition"] == "overall"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for (model, weighting), group in overall.groupby(["model", "weighting"], sort=False):
        group = group.sort_values("decile")
        ax.plot(group["decile"], group["mean_realized_return"], marker="o", label=f"{model} {weighting}")
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Prediction decile")
    ax.set_ylabel("Mean realized next-month return")
    ax.set_title("Overall prediction-decile return profile")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "decile_profile_overall.png", dpi=180)
    plt.close(fig)

    three = profiles[profiles["regime_definition"] == "three_state"]
    for model in MODEL_COLUMNS:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        subset = three[three["model"] == model]
        for (weighting, regime), group in subset.groupby(["weighting", "regime"], sort=False):
            group = group.sort_values("decile")
            ax.plot(group["decile"], group["mean_realized_return"], marker="o", label=f"{regime} {weighting}")
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Prediction decile")
        ax.set_ylabel("Mean realized next-month return")
        ax.set_title(f"Three-state prediction-decile profile: {model}")
        ax.legend(ncol=2)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        slug = "elastic_net" if model == "Elastic Net" else "xgboost"
        fig.savefig(output_dir / f"decile_profile_three_regime_{slug}.png", dpi=180)
        plt.close(fig)

    for weighting in ("EW", "VW"):
        subset = monthly[monthly["weighting"] == weighting].sort_values("formation_date")
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for model, group in subset.groupby("model", sort=False):
            group = group.sort_values("formation_date")
            cumulative = _finite_numeric(group["D10_minus_D1"]).fillna(0.0).cumsum()
            ax.plot(group["formation_date"], cumulative, label=model)
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xlabel("Formation month")
        ax.set_ylabel("Cumulative arithmetic D10-D1 return")
        ax.set_title(f"Cumulative arithmetic long-short diagnostic ({weighting})")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"cumulative_long_short_{weighting.lower()}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(regimes["date"], regimes["vix"], label="VIX")
    ax.plot(regimes["date"], regimes["vix_expanding_q33"], label="Expanding q33")
    ax.plot(regimes["date"], regimes["vix_expanding_q67"], label="Expanding q67")
    for regime in REGIME_3STATE_ORDER:
        mask = regimes["regime_3state"] == regime
        ax.scatter(regimes.loc[mask, "date"], regimes.loc[mask, "vix"], s=10, label=f"{regime} months")
    ax.set_xlabel("Month")
    ax.set_ylabel("VIX")
    ax.set_title("Formation-time VIX with expanding historical terciles")
    ax.legend(ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "vix_three_regimes.png", dpi=180)
    plt.close(fig)


def economic_value_summary_markdown(
    overall: pd.DataFrame,
    binary: pd.DataFrame,
    three: pd.DataFrame,
    rank: pd.DataFrame,
    monotonicity: pd.DataFrame,
) -> str:
    def line(row: pd.Series) -> str:
        return (
            f"- {row['model']} {row['weighting']} {row['regime']}: "
            f"mean monthly D10-D1={row['mean_D10_minus_D1']:.4%}, "
            f"annualized arithmetic={row['annualized_D10_minus_D1']:.2%}, "
            f"descriptive Sharpe={row['descriptive_sharpe']:.2f}, N={int(row['n_months'])}."
        )

    parts = [
        "# Economic Value of Baseline Predictions",
        "",
        "## Baseline specification",
        "Predictors: Size (`log_me`), Book-to-Market (`book_to_market`), and Momentum 12-2 (`mom_12_2`).",
        "Models: Elastic Net and XGBoost. Predictions are frozen OOS forecasts; portfolio analysis does not retrain models.",
        "",
        "## Portfolio construction",
        "Stocks are ranked by predicted next-month return within each formation month and model, exact ties are broken deterministically by PERMNO, and ten approximately equal portfolios are formed. D10-D1 is reported gross of transaction costs in this extension. EW and formation-time lagged-ME VW portfolios are both retained.",
        "",
        "## Overall results",
    ]
    parts.extend(line(row) for _, row in overall.iterrows())
    parts.extend(["", "## Binary VIX regimes"])
    parts.extend(line(row) for _, row in binary.iterrows())
    parts.extend(["", "## Three-state VIX regimes"])
    parts.extend(line(row) for _, row in three.iterrows())
    parts.extend(["", "## Cross-sectional rank performance"])
    three_rank = rank[rank["regime_definition"] == "three_state"]
    for _, row in three_rank.iterrows():
        parts.append(
            f"- {row['model']} {row['regime']}: mean monthly Spearman IC={row['mean_spearman']:.5f}, N={int(row['n_months'])}."
        )
    parts.extend(["", "## Decile-profile evidence"])
    for _, row in monotonicity[monotonicity["regime_definition"] == "three_state"].iterrows():
        parts.append(
            f"- {row['model']} {row['weighting']} {row['regime']}: descriptive decile-return Spearman={row['decile_return_spearman']:.3f}; {int(row['adjacent_increases'])}/9 adjacent increases."
        )
    parts.extend(
        [
            "",
            "## Interpretation",
            "The LOW/MIDDLE/HIGH comparisons above are descriptive conditional-performance estimates. The three-state classification is an extension to, not a replacement for, the original expanding-median binary regime.",
            "",
            "## Limitations",
            "Formal three-way HAC/Newey-West regime-difference inference is not claimed in this stage. Transaction-cost analysis remains the previously implemented binary-regime robustness exercise and is not re-optimized for the three-state extension. Smaller regime samples should be interpreted with their reported month counts.",
            "",
        ]
    )
    return "\n".join(parts)


def _read_canonical_vix_history(panel_path: Path) -> pd.DataFrame:
    """Read only formation date and VIX from the validated canonical panel."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read the canonical modeling panel.") from exc
    schema = set(pq.ParquetFile(panel_path).schema.names)
    date_col = "formation_date" if "formation_date" in schema else "date" if "date" in schema else None
    if date_col is None or "vix" not in schema:
        raise ValueError("Canonical modeling panel must contain a formation date and vix.")
    panel = pd.read_parquet(panel_path, columns=[date_col, "vix"])
    return monthly_vix_history_from_panel(panel)


def run_three_regime_economic_value(repo_root: Path) -> dict[str, Path]:
    """Run the production gross economic-value extension from frozen predictions."""
    prediction_path = repo_root / "results" / "baseline_models" / "baseline_oos_predictions.parquet"
    panel_path = repo_root / "data" / "processed" / "baseline_modeling_panel.parquet"
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Validated production predictions are missing: {prediction_path}. "
            "Do not retrain automatically; restore the saved baseline prediction artifact."
        )
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Validated canonical modeling panel is missing: {panel_path}. "
            "It is required only for the pre-OOS VIX history used by expanding thresholds."
        )

    output_dir = repo_root / "results" / "economic_value"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/8] Loading validated baseline OOS predictions...")
    predictions = pd.read_parquet(prediction_path)
    prediction_columns = MODEL_COLUMNS.values()
    before_predictions = predictions[list(prediction_columns)].copy()

    print("[2/8] Constructing binary and three-state VIX regimes...")
    vix_history = _read_canonical_vix_history(panel_path)
    augmented, regimes = attach_volatility_regimes(predictions, vix_history)

    print("[3/8] Validating volatility-state timing...")
    if not np.allclose(
        before_predictions.to_numpy(float),
        augmented[list(prediction_columns)].to_numpy(float),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    ):
        raise AssertionError("Portfolio evaluation altered saved baseline predictions.")
    validation = regime_validation_table(regimes, augmented)
    if not validation["q33_le_q67_all_valid"].all():
        raise AssertionError("q33 <= q67 validation failed.")

    print("[4/8] Assigning monthly prediction deciles...")
    assignments = build_portfolio_assignments(augmented)

    print("[5/8] Computing equal-weighted portfolios...")
    monthly = monthly_decile_returns(assignments)
    if monthly[monthly["weighting"] == "EW"].empty:
        raise RuntimeError("No equal-weighted portfolio returns were produced.")

    print("[6/8] Computing value-weighted portfolios...")
    if monthly[monthly["weighting"] == "VW"].empty:
        raise RuntimeError("No value-weighted portfolio returns were produced.")

    print("[7/8] Computing rank and regime diagnostics...")
    overall_summary = summarize_portfolios(monthly, "overall")
    binary_summary = summarize_portfolios(monthly, "binary")
    three_summary = summarize_portfolios(monthly, "three_state")
    rank_monthly = monthly_rank_ic(augmented)
    rank_summary = summarize_rank_ic(rank_monthly)
    profiles = decile_return_profiles(monthly)
    monotonicity = monotonicity_diagnostics(profiles)
    transitions = regime_transition_counts(regimes)
    central = central_three_regime_summary(three_summary, rank_summary)

    print("[8/8] Writing economic-value research artifacts...")
    monthly.to_csv(output_dir / "decile_returns_monthly.csv", index=False)
    long_short = monthly[
        [
            "formation_date",
            "realized_return_date",
            "model",
            "regime_binary",
            "regime_3state",
            "weighting",
            "D10",
            "D1",
            "D10_minus_D1",
            "n_stocks",
        ]
    ].rename(columns={"D10": "D10_return", "D1": "D1_return"})
    long_short.to_csv(output_dir / "long_short_returns.csv", index=False)
    assignments.to_csv(output_dir / "portfolio_assignments.csv", index=False)
    overall_summary.to_csv(output_dir / "portfolio_summary_overall.csv", index=False)
    binary_summary.to_csv(output_dir / "portfolio_summary_binary_regime.csv", index=False)
    three_summary.to_csv(output_dir / "portfolio_summary_three_regime.csv", index=False)
    central.to_csv(output_dir / "three_regime_comparison.csv", index=False)
    central.to_csv(output_dir / "three_regime_economic_value_summary.csv", index=False)
    rank_monthly.to_csv(output_dir / "rank_ic_monthly.csv", index=False)
    rank_summary.to_csv(output_dir / "rank_ic_by_volatility_regime.csv", index=False)
    profiles.to_csv(output_dir / "decile_return_profiles.csv", index=False)
    monotonicity.to_csv(output_dir / "monotonicity_diagnostics.csv", index=False)
    transitions.to_csv(output_dir / "regime_transition_counts.csv", index=False)
    validation.to_csv(output_dir / "regime_validation.csv", index=False)
    regimes.to_csv(output_dir / "vix_regime_monthly.csv", index=False)
    write_figures(monthly, profiles, regimes, output_dir)
    summary_text = economic_value_summary_markdown(
        overall_summary,
        binary_summary,
        three_summary,
        rank_summary,
        monotonicity,
    )
    (output_dir / "economic_value_summary.md").write_text(summary_text, encoding="utf-8")

    metadata = pd.DataFrame(
        [
            {
                "prediction_rows": int(len(predictions)),
                "prediction_months": int(augmented["formation_date"].nunique()),
                "vix_history_months": int(len(regimes)),
                "portfolio_assignment_rows": int(len(assignments)),
                "monthly_portfolio_rows": int(len(monthly)),
                "quantile_method": QUANTILE_METHOD,
                "threshold_includes_current_month": True,
                "minimum_history_observations": 1,
                "models_retrained": False,
                "portfolio_results_used_to_define_regimes": False,
                "transaction_costs_applied_in_three_state_extension": False,
            }
        ]
    )
    metadata.to_csv(output_dir / "run_metadata.csv", index=False)

    return {
        "output_dir": output_dir,
        "monthly": output_dir / "decile_returns_monthly.csv",
        "three_regime_summary": output_dir / "three_regime_economic_value_summary.csv",
        "rank_summary": output_dir / "rank_ic_by_volatility_regime.csv",
        "regime_validation": output_dir / "regime_validation.csv",
        "summary": output_dir / "economic_value_summary.md",
    }

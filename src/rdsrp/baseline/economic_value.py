"""Economic-value evaluation for production baseline return forecasts.

The functions in this module consume already-generated out-of-sample predictions.
They do not refit models or alter the production prediction sample.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

MODEL_COLUMNS: Mapping[str, str] = {
    "Elastic Net": "elastic_net_prediction",
    "XGBoost": "xgboost_prediction",
}


def _validate_prediction_inputs(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "permno",
        "regime",
        "actual_next_month_return",
        "me_lag",
        "me_lag_date",
        *MODEL_COLUMNS.values(),
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Portfolio evaluation is missing required columns: {missing}")

    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["me_lag_date"] = pd.to_datetime(frame["me_lag_date"])
    expected = frame["date"] + pd.offsets.MonthEnd(-1)
    if not frame["me_lag_date"].eq(expected).all():
        raise AssertionError("me_lag_date must equal the previous calendar month-end for every row.")
    if frame.duplicated(["permno", "date"]).any():
        raise AssertionError("Prediction panel contains duplicate permno/date rows.")
    monthly_regimes = frame.groupby("date", sort=False)["regime"].nunique(dropna=True)
    if (monthly_regimes != 1).any():
        raise AssertionError("Regime must be unique within each formation month.")
    return frame


def assign_prediction_deciles(values: pd.Series, *, n_portfolios: int = 10) -> pd.Series:
    """Assign tie-preserving percentile portfolios from low (1) to high (n).

    Average percentile ranks keep identical predictions in the same portfolio,
    avoiding arbitrary ordering by row position. Some intermediate portfolios may
    be empty when predictions contain large ties; the extreme portfolios must exist
    for the long-short return to be defined.
    """
    if n_portfolios < 2:
        raise ValueError("n_portfolios must be at least 2.")
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid = numeric.notna() & np.isfinite(numeric)
    if not valid.any():
        return out
    pct = numeric.loc[valid].rank(method="average", pct=True)
    assigned = np.ceil(float(n_portfolios) * pct).clip(1, n_portfolios).astype(int)
    out.loc[valid] = assigned
    return out


def _target_weights(group: pd.DataFrame, portfolio: int, weighting: str) -> dict[int, float]:
    side = group[group["portfolio"] == portfolio]
    if weighting == "VW":
        weights = pd.to_numeric(side["me_lag"], errors="coerce")
        valid = weights.notna() & np.isfinite(weights) & (weights > 0)
        side = side.loc[valid]
        weights = weights.loc[valid]
    elif weighting == "EW":
        weights = pd.Series(1.0, index=side.index)
    else:
        raise ValueError("weighting must be 'EW' or 'VW'.")

    if side.empty:
        return {}
    normalized = weights.to_numpy(float)
    normalized = normalized / normalized.sum()
    return dict(zip(side["permno"].astype(int), normalized, strict=True))


def _weighted_return(group: pd.DataFrame, weights: Mapping[int, float]) -> float:
    if not weights:
        return float("nan")
    returns = group.set_index("permno")["actual_next_month_return"]
    values = []
    wts = []
    for permno, weight in weights.items():
        if permno not in returns.index:
            continue
        value = float(returns.loc[permno])
        if np.isfinite(value):
            values.append(value)
            wts.append(float(weight))
    if not wts:
        return float("nan")
    weight_array = np.asarray(wts, dtype=float)
    weight_array /= weight_array.sum()
    return float(np.dot(weight_array, np.asarray(values, dtype=float)))


def one_way_turnover(
    new_weights: Mapping[int, float],
    previous_state: tuple[Mapping[int, float], Mapping[int, float]] | None,
) -> float:
    """One-way turnover after drifting prior weights through realized returns.

    A newly initiated side has turnover one. Thereafter turnover is one half of
    the L1 distance between new target weights and return-drifted prior weights.
    """
    if previous_state is None:
        return 1.0
    previous_weights, previous_returns = previous_state
    drifted: dict[int, float] = {}
    for permno, weight in previous_weights.items():
        realized = float(previous_returns.get(permno, np.nan))
        if np.isfinite(realized):
            drifted[int(permno)] = float(weight) * (1.0 + realized)
    total = float(sum(drifted.values()))
    if total != 0.0:
        drifted = {permno: weight / total for permno, weight in drifted.items()}
    names = set(new_weights) | set(drifted)
    return float(
        0.5
        * sum(
            abs(float(new_weights.get(permno, 0.0)) - float(drifted.get(permno, 0.0)))
            for permno in names
        )
    )


def monthly_decile_long_short_returns(
    predictions: pd.DataFrame,
    *,
    transaction_cost_bps: float = 50.0,
    n_portfolios: int = 10,
) -> pd.DataFrame:
    """Build monthly 10-1 EW/VW portfolios from each model's OOS predictions.

    Transaction costs are charged at ``transaction_cost_bps`` per dollar of
    one-way turnover on each leg. Total long-short turnover is long turnover plus
    short turnover. Returns remain expressed per one dollar long minus one dollar
    short, matching the gross 10-1 return definition.
    """
    frame = _validate_prediction_inputs(predictions)
    cost_rate = float(transaction_cost_bps) / 10_000.0
    rows: list[dict[str, Any]] = []

    for model, pred_col in MODEL_COLUMNS.items():
        for weighting in ("EW", "VW"):
            previous_long = None
            previous_short = None
            for date, raw_group in frame.groupby("date", sort=True):
                group = raw_group.dropna(subset=[pred_col, "actual_next_month_return"]).copy()
                group = group[
                    np.isfinite(pd.to_numeric(group[pred_col], errors="coerce"))
                    & np.isfinite(pd.to_numeric(group["actual_next_month_return"], errors="coerce"))
                ]
                group["portfolio"] = assign_prediction_deciles(
                    group[pred_col], n_portfolios=n_portfolios
                )
                low_weights = _target_weights(group, 1, weighting)
                high_weights = _target_weights(group, n_portfolios, weighting)
                if not low_weights or not high_weights:
                    raise RuntimeError(
                        f"Cannot form both extreme portfolios for {model}/{weighting}/{pd.Timestamp(date).date()}."
                    )

                low_return = _weighted_return(group, low_weights)
                high_return = _weighted_return(group, high_weights)
                gross = high_return - low_return

                long_turnover = one_way_turnover(high_weights, previous_long)
                short_turnover = one_way_turnover(low_weights, previous_short)
                total_turnover = long_turnover + short_turnover
                transaction_cost = cost_rate * total_turnover
                net = gross - transaction_cost

                realized = dict(
                    zip(
                        group["permno"].astype(int),
                        group["actual_next_month_return"].astype(float),
                        strict=True,
                    )
                )
                previous_long = (high_weights, realized)
                previous_short = (low_weights, realized)

                rows.append(
                    {
                        "date": pd.Timestamp(date),
                        "regime": str(group["regime"].iloc[0]).upper(),
                        "model": model,
                        "weighting": weighting,
                        "n_long": len(high_weights),
                        "n_short": len(low_weights),
                        "gross_long_short_return": gross,
                        "long_turnover": long_turnover,
                        "short_turnover": short_turnover,
                        "total_turnover": total_turnover,
                        "transaction_cost": transaction_cost,
                        "net_long_short_return": net,
                    }
                )
    return pd.DataFrame(rows)


def hac_mean_inference(values: pd.Series | np.ndarray, *, maxlags: int = 6) -> dict[str, float]:
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 2:
        raise ValueError("At least two finite observations are required for HAC inference.")
    mean = float(y.mean())
    residuals = y - mean
    meat = float(np.sum(residuals**2))
    for lag in range(1, min(maxlags + 1, len(y))):
        weight = 1.0 - lag / (maxlags + 1.0)
        meat += 2.0 * weight * float(np.sum(residuals[lag:] * residuals[:-lag]))
    variance = max(meat / (len(y) ** 2), 0.0)
    se = math.sqrt(variance)
    t_stat = mean / se if se > 0 else float("nan")
    p_value = math.erfc(abs(t_stat) / math.sqrt(2.0)) if np.isfinite(t_stat) else float("nan")
    return {"mean": mean, "hac_se": se, "hac_t": t_stat, "p_value": p_value}


def hac_regime_difference(
    values: pd.Series | np.ndarray,
    regimes: pd.Series | np.ndarray,
    *,
    maxlags: int = 6,
) -> dict[str, float]:
    y = np.asarray(values, dtype=float)
    regime_array = np.asarray(regimes).astype(str)
    valid = np.isfinite(y) & np.isin(regime_array, ["HIGH", "LOW"])
    y = y[valid]
    low = (regime_array[valid] == "LOW").astype(float)
    if len(y) < 3 or low.min() == low.max():
        raise ValueError("Both HIGH and LOW observations are required for the regime-difference test.")

    X = np.column_stack([np.ones(len(y)), low])
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    residuals = y - X @ beta
    meat = np.zeros((2, 2), dtype=float)
    for t in range(len(y)):
        meat += residuals[t] ** 2 * np.outer(X[t], X[t])
    for lag in range(1, min(maxlags + 1, len(y))):
        weight = 1.0 - lag / (maxlags + 1.0)
        for t in range(lag, len(y)):
            meat += weight * residuals[t] * residuals[t - lag] * (
                np.outer(X[t], X[t - lag]) + np.outer(X[t - lag], X[t])
            )
    covariance = xtx_inv @ meat @ xtx_inv
    se = float(math.sqrt(max(float(covariance[1, 1]), 0.0)))
    difference = float(beta[1])
    t_stat = difference / se if se > 0 else float("nan")
    p_value = math.erfc(abs(t_stat) / math.sqrt(2.0)) if np.isfinite(t_stat) else float("nan")
    return {
        "high_mean": float(beta[0]),
        "low_mean": float(beta[0] + beta[1]),
        "low_minus_high": difference,
        "hac_se": se,
        "hac_t": t_stat,
        "p_value": p_value,
    }


def summarize_portfolio_performance(
    monthly: pd.DataFrame,
    *,
    hac_lags: int = 6,
) -> pd.DataFrame:
    required = {
        "model",
        "weighting",
        "regime",
        "gross_long_short_return",
        "net_long_short_return",
        "total_turnover",
        "transaction_cost",
    }
    missing = sorted(required.difference(monthly.columns))
    if missing:
        raise ValueError(f"Monthly portfolio table is missing required columns: {missing}")

    rows: list[dict[str, Any]] = []
    for model in MODEL_COLUMNS:
        for weighting in ("EW", "VW"):
            base = monthly[(monthly["model"] == model) & (monthly["weighting"] == weighting)]
            for regime in ("OVERALL", "HIGH", "LOW"):
                group = base if regime == "OVERALL" else base[base["regime"] == regime]
                gross = pd.to_numeric(group["gross_long_short_return"], errors="coerce").dropna()
                net = pd.to_numeric(group["net_long_short_return"], errors="coerce").dropna()
                if gross.empty or net.empty:
                    continue
                gross_inf = hac_mean_inference(gross, maxlags=hac_lags)
                net_inf = hac_mean_inference(net, maxlags=hac_lags)
                net_vol = float(net.std(ddof=1))
                rows.append(
                    {
                        "model": model,
                        "weighting": weighting,
                        "regime": regime,
                        "months": int(len(net)),
                        "gross_mean_monthly": gross_inf["mean"],
                        "gross_annualized_mean": 12.0 * gross_inf["mean"],
                        "gross_hac_t": gross_inf["hac_t"],
                        "gross_p_value": gross_inf["p_value"],
                        "average_total_turnover": float(group["total_turnover"].mean()),
                        "average_transaction_cost": float(group["transaction_cost"].mean()),
                        "net_mean_monthly": net_inf["mean"],
                        "net_annualized_mean": 12.0 * net_inf["mean"],
                        "net_annualized_volatility": math.sqrt(12.0) * net_vol,
                        "net_sharpe": math.sqrt(12.0) * net_inf["mean"] / net_vol
                        if net_vol > 0
                        else float("nan"),
                        "net_hac_t": net_inf["hac_t"],
                        "net_p_value": net_inf["p_value"],
                        "net_fraction_positive": float((net > 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def portfolio_regime_difference_tests(
    monthly: pd.DataFrame,
    *,
    hac_lags: int = 6,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODEL_COLUMNS:
        for weighting in ("EW", "VW"):
            group = monthly[
                (monthly["model"] == model) & (monthly["weighting"] == weighting)
            ].sort_values("date")
            for return_type, column in (
                ("gross", "gross_long_short_return"),
                ("net", "net_long_short_return"),
            ):
                result = hac_regime_difference(group[column], group["regime"], maxlags=hac_lags)
                rows.append(
                    {
                        "model": model,
                        "weighting": weighting,
                        "return_type": return_type,
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def rank_ic_inference(monthly_rank: pd.DataFrame, *, hac_lags: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    within_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []
    for model, group in monthly_rank.groupby("model", sort=False):
        group = group.sort_values("date")
        for regime in ("OVERALL", "HIGH", "LOW"):
            subset = group if regime == "OVERALL" else group[group["regime"] == regime]
            inference = hac_mean_inference(subset["spearman_ic"], maxlags=hac_lags)
            within_rows.append(
                {
                    "model": model,
                    "regime": regime,
                    "months": int(len(subset)),
                    "mean_spearman_ic": inference["mean"],
                    "hac_se": inference["hac_se"],
                    "hac_t": inference["hac_t"],
                    "p_value": inference["p_value"],
                }
            )
        diff = hac_regime_difference(group["spearman_ic"], group["regime"], maxlags=hac_lags)
        difference_rows.append({"model": model, **diff})
    return pd.DataFrame(within_rows), pd.DataFrame(difference_rows)

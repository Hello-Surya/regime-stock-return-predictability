"""Cross-sectional portfolio evaluation utilities.

This module contains deterministic predicted-return sorts, equal- and value-weighted
portfolio returns, long-short summaries, and monthly cross-sectional rank diagnostics.
All grouping is performed within formation month and model.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioDiagnostics:
    skipped_months: pd.DataFrame


def _month_end(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value) + pd.offsets.MonthEnd(0)


def fixed_universe_permnos(
    crsp: pd.DataFrame,
    *,
    formation_date: str | pd.Timestamp,
    top_n: int,
) -> pd.Index:
    """Recover the exact fixed top-N research universe from cached CRSP history.

    The single-stock extraction may contain one additional security retained only for
    individual-stock validation. Ranking the cached formation-month observations by
    market equity and PERMNO exactly recovers the top-N formation universe and removes
    any such forced addition unless it independently belongs to the top N.
    """
    if top_n < 1:
        raise ValueError("top_n must be positive.")
    required = {"date", "permno", "market_equity"}
    missing = required.difference(crsp.columns)
    if missing:
        raise ValueError(f"CRSP data missing fixed-universe columns: {sorted(missing)}")

    frame = crsp.copy()
    frame["date"] = pd.to_datetime(frame["date"]) + pd.offsets.MonthEnd(0)
    date = _month_end(formation_date)
    formation = frame.loc[frame["date"] == date, ["permno", "market_equity"]].copy()
    formation["market_equity"] = pd.to_numeric(formation["market_equity"], errors="coerce")
    formation = formation.dropna(subset=["permno", "market_equity"])
    formation = formation[formation["market_equity"] > 0]
    formation = formation.sort_values(
        ["market_equity", "permno"], ascending=[False, True], kind="mergesort"
    )
    if len(formation) < top_n:
        raise RuntimeError(
            f"Cached CRSP data contain only {len(formation)} valid formation-month securities; "
            f"cannot reconstruct the configured top-{top_n} universe."
        )
    return pd.Index(formation.head(top_n)["permno"].to_numpy(), name="permno")


def restrict_to_fixed_universe(
    crsp: pd.DataFrame,
    *,
    formation_date: str | pd.Timestamp,
    top_n: int,
) -> pd.DataFrame:
    permnos = fixed_universe_permnos(crsp, formation_date=formation_date, top_n=top_n)
    return crsp[crsp["permno"].isin(permnos)].copy()


def _group_columns(frame: pd.DataFrame, date_col: str, model_col: str | None) -> list[str]:
    columns = [date_col]
    if model_col is not None and model_col in frame.columns:
        columns.append(model_col)
    return columns


def assign_deciles(
    df: pd.DataFrame,
    prediction_col: str = "prediction",
    date_col: str = "date",
    *,
    model_col: str | None = "model",
    id_col: str = "permno",
    n_portfolios: int = 10,
) -> pd.DataFrame:
    """Assign deterministic approximately equal-sized predicted-return portfolios.

    Within each formation month and model, observations are sorted by predicted return
    and then by PERMNO to break exact prediction ties deterministically. Ordinal ranks
    are mapped to portfolios using ``floor((rank-1) * K / N) + 1``. A group with fewer
    than ``K`` valid predictions is left unassigned rather than silently producing fewer
    portfolios.
    """
    if n_portfolios < 2:
        raise ValueError("n_portfolios must be at least 2.")
    required = {date_col, prediction_col}
    if model_col is not None and model_col in df.columns:
        required.add(model_col)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Prediction data missing decile columns: {sorted(missing)}")

    out = df.copy()
    tie_col = id_col
    if id_col not in out.columns:
        tie_col = "__tie_id__"
        out[tie_col] = np.arange(len(out), dtype=int)
    out[date_col] = pd.to_datetime(out[date_col])
    out["decile"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["decile_status"] = "insufficient_securities"

    groups = _group_columns(out, date_col, model_col)
    grouped = out.groupby(groups, sort=True, dropna=False)
    for _, idx in grouped.groups.items():
        idx = pd.Index(idx)
        block = out.loc[idx]
        valid = block[prediction_col].notna() & block[tie_col].notna()
        n_valid = int(valid.sum())
        if n_valid < n_portfolios:
            continue
        ordered = block.loc[valid].sort_values(
            [prediction_col, tie_col], ascending=[True, True], kind="mergesort"
        )
        ordinal = np.arange(1, n_valid + 1, dtype=int)
        deciles = np.floor((ordinal - 1) * n_portfolios / n_valid).astype(int) + 1
        out.loc[ordered.index, "decile"] = pd.array(deciles, dtype="Int64")
        out.loc[idx, "decile_status"] = "assigned"

    if "__tie_id__" in out.columns:
        out = out.drop(columns="__tie_id__")
    return out


def skipped_decile_months(
    assignments: pd.DataFrame,
    *,
    date_col: str = "formation_date",
    model_col: str = "model",
) -> pd.DataFrame:
    if assignments.empty:
        return pd.DataFrame(columns=[date_col, model_col, "reason", "n_stocks"])
    rows: list[dict[str, object]] = []
    for keys, block in assignments.groupby([date_col, model_col], sort=True):
        if block["decile"].notna().any():
            continue
        date, model = keys
        rows.append(
            {
                date_col: date,
                model_col: model,
                "reason": "fewer_than_10_valid_predictions",
                "n_stocks": int(block["prediction"].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def prepare_portfolio_assignments(
    predictions: pd.DataFrame,
    *,
    weight_col: str = "me_lag",
) -> tuple[pd.DataFrame, PortfolioDiagnostics]:
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
    frame = assign_deciles(frame, date_col="formation_date")
    skipped = skipped_decile_months(frame)
    frame["weighting_variable"] = pd.to_numeric(frame[weight_col], errors="coerce")
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
    ]
    for optional in ("value_weight_date", "train_feature_end_date", "train_target_end_date"):
        if optional in frame.columns:
            columns.append(optional)
    if "ticker" not in frame.columns:
        frame["ticker"] = pd.NA
    return frame[columns].copy(), PortfolioDiagnostics(skipped_months=skipped)


def _weighted_return(values: pd.Series, weights: pd.Series) -> tuple[float, int]:
    x = pd.to_numeric(values, errors="coerce").to_numpy(float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(w) & (w > 0)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return float("nan"), 0
    wv = w[valid]
    xv = x[valid]
    return float(np.average(xv, weights=wv)), n_valid


def compute_decile_returns(
    assignments: pd.DataFrame,
    *,
    return_col: str = "actual_next_month_return",
    weight_col: str = "weighting_variable",
) -> pd.DataFrame:
    required = {
        "formation_date",
        "realized_return_date",
        "model",
        "regime",
        "decile",
        return_col,
        weight_col,
    }
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(f"Portfolio assignments missing columns: {sorted(missing)}")

    valid = assignments[assignments["decile"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    monthly_keys = ["formation_date", "realized_return_date", "model", "regime"]
    rows: list[dict[str, object]] = []
    for keys, month in valid.groupby(monthly_keys, sort=True, dropna=False):
        formation_date, realized_return_date, model, regime = keys
        total_stocks = int(month["permno"].nunique()) if "permno" in month else int(len(month))
        for weighting in ("EW", "VW"):
            row: dict[str, object] = {
                "formation_date": formation_date,
                "realized_return_date": realized_return_date,
                "model": model,
                "regime": regime,
                "weighting": weighting,
                "n_stocks": total_stocks,
            }
            for decile in range(1, 11):
                block = month[month["decile"] == decile]
                row[f"D{decile}_n"] = int(len(block))
                if weighting == "EW":
                    returns = pd.to_numeric(block[return_col], errors="coerce")
                    finite = returns[np.isfinite(returns.to_numpy(float))]
                    row[f"D{decile}"] = float(finite.mean()) if len(finite) else float("nan")
                    row[f"D{decile}_n_weight"] = int(len(finite))
                else:
                    value, n_weight = _weighted_return(block[return_col], block[weight_col])
                    row[f"D{decile}"] = value
                    row[f"D{decile}_n_weight"] = n_weight
            d1 = row["D1"]
            d10 = row["D10"]
            row["D10_minus_D1"] = (
                float(d10) - float(d1)
                if np.isfinite(float(d10)) and np.isfinite(float(d1))
                else float("nan")
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["formation_date", "model", "weighting"]
    ).reset_index(drop=True)


def long_short_returns(decile_returns: pd.DataFrame) -> pd.DataFrame:
    if decile_returns.empty:
        return pd.DataFrame(
            columns=[
                "formation_date",
                "realized_return_date",
                "model",
                "regime",
                "weighting",
                "D10_return",
                "D1_return",
                "D10_minus_D1",
                "n_stocks",
            ]
        )
    out = decile_returns[
        [
            "formation_date",
            "realized_return_date",
            "model",
            "regime",
            "weighting",
            "D10",
            "D1",
            "D10_minus_D1",
            "n_stocks",
            "D10_n_weight",
            "D1_n_weight",
        ]
    ].copy()
    return out.rename(columns={"D10": "D10_return", "D1": "D1_return"})


def _summary_row(block: pd.DataFrame, model: str, regime: str, weighting: str) -> dict[str, object]:
    spread = pd.to_numeric(block["D10_minus_D1"], errors="coerce")
    valid = block.loc[np.isfinite(spread.to_numpy(float))].copy()
    spread = pd.to_numeric(valid["D10_minus_D1"], errors="coerce")
    n = int(len(valid))
    mean = float(spread.mean()) if n else float("nan")
    sd = float(spread.std(ddof=1)) if n > 1 else float("nan")
    annualized_mean = 12.0 * mean if np.isfinite(mean) else float("nan")
    annualized_vol = sqrt(12.0) * sd if np.isfinite(sd) else float("nan")
    sharpe = sqrt(12.0) * mean / sd if np.isfinite(sd) and sd > 0 else float("nan")
    return {
        "model": model,
        "regime": regime,
        "weighting": weighting,
        "n_months": n,
        "average_D1_monthly_return": float(valid["D1_return"].mean()) if n else float("nan"),
        "average_D10_monthly_return": float(valid["D10_return"].mean()) if n else float("nan"),
        "mean_monthly_D10_minus_D1": mean,
        "sd_monthly_D10_minus_D1": sd,
        "annualized_arithmetic_mean": annualized_mean,
        "annualized_volatility": annualized_vol,
        "descriptive_sharpe": sharpe,
        "fraction_positive": float((spread > 0).mean()) if n else float("nan"),
    }


def summarize_portfolios(long_short: pd.DataFrame) -> pd.DataFrame:
    if long_short.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (model, weighting), all_block in long_short.groupby(["model", "weighting"], sort=True):
        rows.append(_summary_row(all_block, str(model), "OVERALL", str(weighting)))
        for regime in ("HIGH", "LOW"):
            block = all_block[all_block["regime"] == regime]
            rows.append(_summary_row(block, str(model), regime, str(weighting)))
    order = {"OVERALL": 0, "HIGH": 1, "LOW": 2}
    out = pd.DataFrame(rows)
    out["_regime_order"] = out["regime"].map(order)
    return out.sort_values(["model", "weighting", "_regime_order"]).drop(
        columns="_regime_order"
    ).reset_index(drop=True)


def regime_portfolio_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    cols = [
        "model",
        "regime",
        "weighting",
        "mean_monthly_D10_minus_D1",
        "annualized_arithmetic_mean",
        "annualized_volatility",
        "descriptive_sharpe",
        "fraction_positive",
        "n_months",
    ]
    return summary[cols].copy()


def monthly_rank_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
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
    for keys, block in frame.groupby(["formation_date", "model"], sort=True):
        formation_date, model = keys
        clean = block.dropna(subset=["prediction", "actual_next_month_return"]).copy()
        if len(clean) < 2:
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
                "formation_date": formation_date,
                "realized_return_date": realized_dates[0] if len(realized_dates) else pd.NaT,
                "model": model,
                "regime": regimes[0] if len(regimes) else pd.NA,
                "n_stocks": int(len(clean)),
                "spearman": spearman,
            }
        )
    return pd.DataFrame(rows).sort_values(["formation_date", "model"]).reset_index(drop=True)


def summarize_rank_metrics(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for model, all_block in monthly.groupby("model", sort=True):
        for regime, block in (
            ("OVERALL", all_block),
            ("HIGH", all_block[all_block["regime"] == "HIGH"]),
            ("LOW", all_block[all_block["regime"] == "LOW"]),
        ):
            values = pd.to_numeric(block["spearman"], errors="coerce")
            values = values[np.isfinite(values.to_numpy(float))]
            n = int(len(values))
            rows.append(
                {
                    "model": model,
                    "regime": regime,
                    "n_months": n,
                    "mean_spearman": float(values.mean()) if n else float("nan"),
                    "median_spearman": float(values.median()) if n else float("nan"),
                    "sd_spearman": float(values.std(ddof=1)) if n > 1 else float("nan"),
                    "fraction_positive": float((values > 0).mean()) if n else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def validate_portfolio_outputs(
    predictions: pd.DataFrame,
    assignments: pd.DataFrame,
    decile_returns: pd.DataFrame,
) -> None:
    if predictions.empty:
        raise ValueError("Cross-sectional prediction panel is empty.")
    p = predictions.copy()
    for col in (
        "formation_date",
        "realized_return_date",
        "train_feature_end_date",
        "train_target_end_date",
    ):
        p[col] = pd.to_datetime(p[col])
    if not (p["realized_return_date"] > p["formation_date"]).all():
        raise AssertionError("Target timing failed: realized return must be after formation.")
    if not (p["train_feature_end_date"] < p["formation_date"]).all():
        raise AssertionError("Training feature dates are not strictly historical.")
    if not (p["train_target_end_date"] <= p["formation_date"]).all():
        raise AssertionError("Training targets were not observable by formation.")
    if "value_weight_date" in p:
        value_date = pd.to_datetime(p["value_weight_date"])
        finite_weight = np.isfinite(pd.to_numeric(p["me_lag"], errors="coerce").to_numpy(float))
        if finite_weight.any() and not (value_date[finite_weight] <= p.loc[finite_weight, "formation_date"]).all():
            raise AssertionError("Value weights contain post-formation information.")

    valid_months = assignments[assignments["decile"].notna()].groupby(
        ["formation_date", "model"], sort=True
    )["decile"].apply(lambda s: set(int(x) for x in s.dropna()))
    for key, deciles in valid_months.items():
        if deciles != set(range(1, 11)):
            raise AssertionError(f"Malformed decile month {key}: {sorted(deciles)}")

    if not decile_returns.empty:
        calc = decile_returns["D10"] - decile_returns["D1"]
        mask = calc.notna() & decile_returns["D10_minus_D1"].notna()
        if not np.allclose(calc[mask], decile_returns.loc[mask, "D10_minus_D1"]):
            raise AssertionError("D10-D1 arithmetic validation failed.")

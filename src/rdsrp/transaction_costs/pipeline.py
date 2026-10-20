"""Transaction costs on authoritative M8 assignments, with M9 HAC inference.

Does NOT construct deciles, refit models, access WRDS or mutate upstream inputs.
Dates: formation_date=t; assignment return is realized at t+1; turnover for
formation t uses only *previous* formation t-1 holdings and returns realized at t.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rdsrp.inference.hac import BASELINE_HAC_LAG
from rdsrp.inference.pipeline import (
    long_short_hac_inference,
    long_short_regime_differences,
)

MODELS = ("Elastic Net", "XGBoost")
WEIGHTINGS = ("EW", "VW")
REGIMES = ("LOW", "MIDDLE", "HIGH")
PRIMARY_BPS = (0, 50)
# Official M8/M9 persisted contracts; never substitute historic binary cost inputs.
REQUIRED_ASSIGNMENTS = (
    "formation_date", "realized_return_date", "permno", "model", "decile",
    "realized_next_month_return", "weighting_variable", "regime_binary", "regime_3state",
)
GROSS_KEYS = ["formation_date", "model", "weighting"]
FLOAT_TOL = dict(rtol=1e-10, atol=1e-12)


def bps_to_decimal(bps: float) -> float:
    value = float(bps)
    if not np.isfinite(value) or value < 0:
        raise ValueError("Cost basis points must be finite and nonnegative")
    return value / 10000.0


def _assert_normalized(weights: dict[int, float], *, name: str) -> None:
    if not weights:
        raise AssertionError(f"{name} must be nonempty")
    arr = np.asarray(list(weights.values()), dtype=float)
    if not np.isfinite(arr).all() or (arr < 0).any():
        raise AssertionError(f"{name} has nonfinite or negative weights")
    if not np.isclose(arr.sum(), 1.0, rtol=1e-12, atol=1e-12):
        raise AssertionError(f"{name} does not sum to one: {arr.sum()}")


def target_weights(group: pd.DataFrame, weighting: str) -> dict[int, float]:
    """Exactly mirror M8 _decile_return eligible stocks and target weights."""
    if group.empty:
        raise ValueError("Empty decile cannot be reconstructed")
    if group["permno"].duplicated().any():
        raise AssertionError("Duplicate PERMNO in a decile")
    r = pd.to_numeric(group["realized_next_month_return"], errors="coerce")
    valid = r.notna() & np.isfinite(r)
    if weighting == "EW":
        amounts = np.ones(int(valid.sum()), dtype=float)
    elif weighting == "VW":
        me = pd.to_numeric(group["weighting_variable"], errors="coerce")
        valid &= me.notna() & np.isfinite(me) & (me > 0)
        amounts = me.loc[valid].to_numpy(float)
    else:
        raise ValueError("weighting must be EW or VW")
    if not valid.any():
        raise AssertionError(f"No M8-eligible constituents for {weighting} decile")
    # Scale before summing to avoid overflow on very large market equity.
    if not np.isfinite(amounts).all() or (amounts <= 0).any():
        raise AssertionError("Invalid positive weight inputs")
    scaled = amounts / amounts.max()
    weights = scaled / scaled.sum()
    result = dict(zip(group.loc[valid, "permno"].astype(int), weights.tolist(), strict=True))
    _assert_normalized(result, name="target")
    return result


def drift_weights(previous_targets: dict[int, float], previous_realizations: dict[int, float]) -> dict[int, float]:
    """Carry t-1 weights through returns realized in t, before t formation."""
    _assert_normalized(previous_targets, name="previous target")
    missing = set(previous_targets) - set(previous_realizations)
    if missing:
        raise AssertionError(f"Previous positions missing realized returns: {len(missing)}")
    prior = np.asarray([previous_targets[k] for k in previous_targets], dtype=float)
    returns = np.asarray([previous_realizations[k] for k in previous_targets], dtype=float)
    if not np.isfinite(returns).all() or (returns < -1.0).any():
        raise AssertionError("Invalid previous holding period returns for drift")
    post = prior * (1.0 + returns)
    total = float(post.sum())
    if not np.isfinite(total) or total <= 0:
        raise AssertionError("Previous leg lost all capital; drift normalization undefined")
    out = {k: float(v / total) for k, v in zip(previous_targets, post, strict=True)}
    _assert_normalized(out, name="pretrade")
    return out


def calculate_turnover(target: dict[int, float], pretrade: dict[int, float] | None) -> float:
    """One-sided rebalancing turnover; initial cash-funded leg costs 1.0."""
    _assert_normalized(target, name="target")
    if pretrade is None:
        return 1.0  # initial formation from cash is 100% purchases
    _assert_normalized(pretrade, name="pretrade")
    union = set(target) | set(pretrade)
    result = 0.5 * math.fsum(abs(target.get(i, 0.0) - pretrade.get(i, 0.0)) for i in union)
    if not np.isfinite(result) or not -1e-12 <= result <= 1.0 + 1e-12:
        raise AssertionError("One-way turnover is outside [0,1]")
    return float(result)


def _weighted_return(group: pd.DataFrame, weights: dict[int, float]) -> float:
    returns = group.set_index("permno")["realized_next_month_return"].astype(float)
    return float(math.fsum(weight * float(returns.loc[permno]) for permno, weight in weights.items()))


def _guard_unique_regimes(rows: pd.DataFrame) -> tuple[str, str, pd.Timestamp]:
    b = rows["regime_binary"].astype(str).str.upper().unique()
    s = rows["regime_3state"].astype(str).str.upper().unique()
    d = pd.to_datetime(rows["realized_return_date"]).unique()
    if len(b) != 1 or b[0] not in ("LOW", "HIGH") or len(s) != 1 or s[0] not in REGIMES or len(d) != 1:
        raise AssertionError("M8 month has inconsistent binary/three-state regime or realized date")
    return b[0], s[0], pd.Timestamp(d[0])


def validate_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_ASSIGNMENTS) - set(assignments))
    if missing:
        raise ValueError(f"Missing frozen M8 assignment columns: {missing}")
    work = assignments.copy()
    for c in ("formation_date", "realized_return_date"):
        work[c] = pd.to_datetime(work[c], errors="raise")
        if work[c].isna().any() or not work[c].eq(work[c].dt.to_period("M").dt.to_timestamp("M")).all():
            raise AssertionError(f"{c} must be a valid month-end")
    if not work["realized_return_date"].eq(work["formation_date"] + pd.offsets.MonthEnd(1)).all():
        raise AssertionError("M8 realized_return_date is not formation_date + 1 month")
    for c in ("permno", "decile"):
        numeric = pd.to_numeric(work[c], errors="raise")
        if numeric.isna().any() or not np.isfinite(numeric).all() or (numeric != numeric.astype(int)).any():
            raise AssertionError(f"Invalid {c}")
        work[c] = numeric.astype(int)
    if not work["decile"].between(1, 10).all():
        raise AssertionError("Deciles must be in [1,10]")
    if work.duplicated(["formation_date", "model", "permno"]).any():
        raise AssertionError("Duplicate M8 assignments")
    if set(work["model"]) != set(MODELS):
        raise AssertionError("The frozen two-model set is not present")
    for c in ("regime_binary", "regime_3state"):
        work[c] = work[c].astype(str).str.upper()
    work["realized_next_month_return"] = pd.to_numeric(work["realized_next_month_return"], errors="raise")
    if not np.isfinite(work["realized_next_month_return"]).all():
        raise AssertionError("M8 assignments have invalid realized returns")
    for model in MODELS:
        months = work.loc[work["model"] == model, "formation_date"].drop_duplicates().sort_values().reset_index(drop=True)
        if len(months) < 2 or not months.equals(work["formation_date"].drop_duplicates().sort_values().reset_index(drop=True)):
            raise AssertionError("Frozen months differ across models or contain too few observations")
        gaps = months.dt.to_period("M").astype(int).diff().iloc[1:]
        if not gaps.eq(1).all():
            raise AssertionError("Nonconsecutive formation months: can't infer prior-rebalance drift")
        counts = work.loc[work["model"] == model].groupby("formation_date")["decile"].nunique()
        if not counts.eq(10).all():
            raise AssertionError("Missing decile in a frozen month")
    return work


def reconstruct_portfolios(assignments: pd.DataFrame, *, audit_extreme_only: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return M8-weight-replicated monthly gross, decile turnovers, and audit weights.

    The turnover table covers D1..D10. Audit weights cover D1/D10 by default
    to limit local disk footprint; they include old-only exits and new entrants.
    """
    work = validate_assignments(assignments)
    month_rows: list[dict[str, Any]] = []
    turnover_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    for model in MODELS:
        by_month = work[work["model"] == model].groupby("formation_date", sort=True)
        for weighting in WEIGHTINGS:
            previous: dict[int, tuple[dict[int, float], dict[int, float]]] = {}
            for date, group in by_month:
                b, s, realized_date = _guard_unique_regimes(group)
                if realized_date != pd.Timestamp(date) + pd.offsets.MonthEnd(1):
                    raise AssertionError("Realization timing discrepancy")
                month: dict[str, Any] = dict(formation_date=date, realized_return_date=realized_date,
                    model=model, weighting=weighting, regime_binary=b, regime_3state=s)
                for decile in range(1, 11):
                    subset = group[group["decile"] == decile]
                    target = target_weights(subset, weighting)
                    pre = None if decile not in previous else drift_weights(*previous[decile])
                    turnover = calculate_turnover(target, pre)
                    if not 0 <= turnover <= 1 + 1e-12:
                        raise AssertionError("Invalid turnover")
                    month[f"D{decile}"] = _weighted_return(subset, target)
                    month[f"n_D{decile}"] = len(target)
                    month[f"D{decile}_turnover"] = turnover
                    turnover_rows.append(dict(formation_date=date, model=model, weighting=weighting,
                        decile=f"D{decile}", turnover=turnover, n_stocks=len(target),
                        regime_binary=b, regime_3state=s, initial_formation=pre is None))
                    if not audit_extreme_only or decile in (1, 10):
                        tickers = (subset.set_index("permno")["ticker"].to_dict()
                                   if "ticker" in subset else {})
                        for permno in sorted(set(target) | set(pre or {})):
                            cur = target.get(permno, 0.0)
                            prv = (pre or {}).get(permno, 0.0)
                            weight_rows.append(dict(formation_date=date, permno=permno,
                                ticker=tickers.get(permno), model=model, decile=f"D{decile}",
                                weighting=weighting, target_weight=cur, pretrade_weight=prv,
                                absolute_weight_change=abs(cur-prv), regime_binary=b,
                                regime_3state=s, initial_formation=pre is None))
                    # The *current* realized t+1 returns only feed the *next* month's pretrade weights.
                    current_returns = subset.set_index("permno")["realized_next_month_return"].astype(float).to_dict()
                    previous[decile] = (target, current_returns)
                month["D10_minus_D1"] = month["D10"] - month["D1"]
                month["n_stocks"] = sum(month[f"n_D{i}"] for i in range(1,11))
                month_rows.append(month)
    monthly = pd.DataFrame(month_rows).sort_values(GROSS_KEYS, kind="mergesort").reset_index(drop=True)
    return monthly, pd.DataFrame(turnover_rows), pd.DataFrame(weight_rows)


def _normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ("formation_date", "realized_return_date"):
        if column in out:
            out[column] = pd.to_datetime(out[column], errors="raise")
    return out


def validate_gross_replication(reconstructed: pd.DataFrame, stored: pd.DataFrame) -> None:
    """Protect ALL ten M8 gross deciles, spread, counts, regimes and timing."""
    gross = _normalize_dates(stored)
    if gross.duplicated(GROSS_KEYS).any() or reconstructed.duplicated(GROSS_KEYS).any():
        raise AssertionError("Duplicate gross portfolio keys")
    if len(gross) != len(reconstructed):
        raise AssertionError("Gross portfolio row-count mismatch")
    joined = reconstructed.merge(gross, on=GROSS_KEYS, how="outer", validate="one_to_one",
                                 suffixes=("_new", "_M8"), indicator=True)
    if not joined["_merge"].eq("both").all():
        raise AssertionError("Missing or extra M8 portfolio month")
    for col in [*(f"D{i}" for i in range(1, 11)), "D10_minus_D1"]:
        lhs = joined[f"{col}_new"].to_numpy(float)
        rhs = joined[f"{col}_M8"].to_numpy(float)
        if not np.isfinite(lhs).all() or not np.isfinite(rhs).all() or not np.allclose(lhs, rhs, **FLOAT_TOL):
            delta = np.nanmax(abs(lhs-rhs))
            raise AssertionError(f"M8 gross {col} replication FAILED; maximum diff={delta}")
    for col in ("realized_return_date", "regime_binary", "regime_3state"):
        if not joined[f"{col}_new"].eq(joined[f"{col}_M8"]).all():
            raise AssertionError(f"M8 {col} changed")
    for c in [*(f"n_D{i}" for i in range(1, 11)), "n_stocks"]:
        if not np.array_equal(joined[f"{c}_new"].to_numpy(int), joined[f"{c}_M8"].to_numpy(int)):
            raise AssertionError(f"M8 {c} count has changed")


def validate_authoritative_long_short(monthly: pd.DataFrame, saved_ls: pd.DataFrame) -> None:
    stored = _normalize_dates(saved_ls)
    join = monthly.merge(stored, on=GROSS_KEYS, validate="one_to_one", how="outer",
                         suffixes=("_new", "_M8"), indicator=True)
    if len(join) != len(monthly) or not join["_merge"].eq("both").all():
        raise AssertionError("M8 long_short_returns.csv key mismatch")
    for left, right in (("D1", "D1_return"), ("D10", "D10_return"),
                        ("D10_minus_D1", "D10_minus_D1")):
        a = join[f"{left}_new"].to_numpy(float) if left == "D10_minus_D1" else join[left].to_numpy(float)
        b = join[f"{right}_M8"].to_numpy(float) if left=="D10_minus_D1" else join[right].to_numpy(float)
        if not np.allclose(a, b, **FLOAT_TOL):
            raise AssertionError(f"M8 authoritative gross {left} mismatch")
    for c in ("regime_binary", "regime_3state", "realized_return_date"):
        if not join[f"{c}_new"].eq(join[f"{c}_M8"]).all():
            raise AssertionError(f"M8 long short {c} mismatch")


def apply_cost_scenarios(monthly: pd.DataFrame, cost_scenarios: tuple[float, ...] = PRIMARY_BPS) -> pd.DataFrame:
    out: list[pd.DataFrame] = []
    for bps in cost_scenarios:
        rate = bps_to_decimal(bps)
        work = monthly[["formation_date", "realized_return_date", "model", "weighting",
                        "regime_binary", "regime_3state", "D1", "D10", "D10_minus_D1",
                        "D1_turnover", "D10_turnover"]].copy()
        work["transaction_cost_bps"] = bps
        work = work.rename(columns={"D1": "D1_gross", "D10": "D10_gross",
                                    "D10_minus_D1": "D10_minus_D1_gross"})
        work["D1_cost"] = work["D1_turnover"] * rate
        work["D10_cost"] = work["D10_turnover"] * rate
        work["D1_net"] = work["D1_gross"] - work["D1_cost"] # long-only D1 performance
        work["D10_net"] = work["D10_gross"] - work["D10_cost"]
        work["D1_short_leg_net_contribution"] = -work["D1_gross"] - work["D1_cost"]
        work["long_short_total_turnover"] = work["D10_turnover"] + work["D1_turnover"]
        work["long_short_cost"] = work["D10_cost"] + work["D1_cost"]
        work["D10_minus_D1_net"] = work["D10_minus_D1_gross"] - work["long_short_cost"]
        if (work[["D1_cost", "D10_cost", "long_short_cost"]] < 0).any().any():
            raise AssertionError("Negative cost")
        if not np.allclose(work["D10_net"]+work["D1_short_leg_net_contribution"],
                           work["D10_minus_D1_net"], **FLOAT_TOL):
            raise AssertionError("Long short leg accounting mismatch")
        if bps == 0 and not np.allclose(work["D10_minus_D1_net"], work["D10_minus_D1_gross"], rtol=0, atol=0):
            raise AssertionError("Zero-cost replication mismatch")
        if (work["D10_minus_D1_net"] > work["D10_minus_D1_gross"] + 1e-12).any():
            raise AssertionError("Net spread greater than gross")
        out.append(work)
    return pd.concat(out, ignore_index=True)


def regime_groups(frame: pd.DataFrame):
    for definition, column, regimes in (
        ("overall", None, ("OVERALL",)),
        ("binary", "regime_binary", ("LOW", "HIGH")),
        ("three_state", "regime_3state", REGIMES),
    ):
        for regime in regimes:
            subset = frame if column is None else frame[frame[column] == regime]
            yield definition, regime, subset


def _turnover_summary(monthly: pd.DataFrame, turnovers: pd.DataFrame) -> pd.DataFrame:
    longshort = monthly[["formation_date", "model", "weighting", "regime_binary", "regime_3state"]].copy()
    longshort["decile"] = "D10-D1"
    longshort["turnover"] = monthly["D10_turnover"] + monthly["D1_turnover"]
    merged = pd.concat([turnovers, longshort], ignore_index=True)
    rows = []
    for (model, weighting, leg), sub in merged.groupby(["model", "weighting", "decile"], sort=True):
        for definition, regime, part in regime_groups(sub):
            a = part["turnover"].to_numpy(float)
            if len(a) == 0:
                raise AssertionError("Empty regime turnover group")
            rows.append(dict(model=model, weighting=weighting, decile=leg,
                regime_definition=definition, regime=regime, n_months=len(a),
                mean_turnover=float(np.mean(a)), median_turnover=float(np.median(a)),
                std_turnover=float(np.std(a, ddof=1)) if len(a)>1 else float("nan"),
                p25=float(np.percentile(a,25)), p75=float(np.percentile(a,75)),
                p95=float(np.percentile(a,95)), maximum=float(np.max(a))))
    return pd.DataFrame(rows)


def _transaction_summaries(net: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    be_rows = []
    for (model, weighting, bps), sub in net.groupby(["model", "weighting", "transaction_cost_bps"], sort=True):
        for definition, regime, part in regime_groups(sub):
            gross = part["D10_minus_D1_gross"].to_numpy(float)
            nret = part["D10_minus_D1_net"].to_numpy(float)
            mean_g = float(gross.mean()); mean_n = float(nret.mean())
            turn = float(part["long_short_total_turnover"].mean())
            drag = float(part["long_short_cost"].mean())
            std = float(np.std(nret, ddof=1)) if len(nret)>1 else float("nan")
            rows.append(dict(model=model, weighting=weighting, regime_definition=definition,
                regime=regime, transaction_cost_bps=bps, n_months=len(part),
                mean_gross_D10_minus_D1=mean_g, mean_total_turnover=turn,
                mean_transaction_cost=drag, mean_net_D10_minus_D1=mean_n,
                annualized_gross_arithmetic=12*mean_g, annualized_net_arithmetic=12*mean_n,
                annualized_net_volatility=math.sqrt(12)*std,
                descriptive_net_sharpe=math.sqrt(12)*mean_n/std if std>0 else float("nan"),
                fraction_net_positive=float(np.mean(nret>0)),
                cost_attrition=mean_g-mean_n,
                cost_attrition_ratio=(mean_g-mean_n)/abs(mean_g) if abs(mean_g)>1e-10 else float("nan")))
            if bps==50:
                be_rate=mean_g/turn if turn>0 and mean_g>0 and np.isfinite(turn) else float("nan")
                be_rows.append(dict(model=model,weighting=weighting,regime_definition=definition,
                    regime=regime, n_months=len(part), mean_gross_spread=mean_g,
                    mean_traded_notional=turn, breakeven_cost_decimal=be_rate,
                    breakeven_cost_bps=be_rate*10000 if np.isfinite(be_rate) else float("nan")))
    return pd.DataFrame(rows), pd.DataFrame(be_rows)


def _compare_gross_m9(recalc: pd.DataFrame, stored: pd.DataFrame, key: list[str], numeric: list[str]) -> None:
    actual = recalc.merge(stored, on=key, validate="one_to_one", how="outer", suffixes=("_new","_M9"),indicator=True)
    if len(actual)!=len(stored) or not actual["_merge"].eq("both").all():
        raise AssertionError("M9 inference rows/keys have changed")
    for c in numeric:
        a=actual[f"{c}_new"].to_numpy(float); b=actual[f"{c}_M9"].to_numpy(float)
        if not np.allclose(a,b,rtol=1e-9,atol=1e-11,equal_nan=True):
            raise AssertionError(f"M9 gross {c} differs: stop")


def _plot_summaries(summary: pd.DataFrame, net: pd.DataFrame, output_dir: Path) -> None:
    view=summary[(summary["regime_definition"]=="overall") & (summary["transaction_cost_bps"]==50)]
    fig, ax=plt.subplots(figsize=(9,5)); x=np.arange(len(view)); width=.36
    ax.bar(x-width/2,view["mean_gross_D10_minus_D1"],width,label="Gross")
    ax.bar(x+width/2,view["mean_net_D10_minus_D1"],width,label="Net: 50 bps")
    ax.set_xticks(x,[f"{m} {w}" for m,w in zip(view.model,view.weighting,strict=True)])
    ax.axhline(0,color="black",linewidth=.8); ax.set_ylabel("Mean monthly D10-D1")
    ax.legend();fig.tight_layout();fig.savefig(output_dir/"gross_vs_net_long_short.png",dpi=180);plt.close(fig)
    three=summary[(summary["regime_definition"]=="three_state") & (summary["transaction_cost_bps"]==50)]
    fig,ax=plt.subplots(figsize=(9,5)); x=np.arange(3); width=.2
    for k,((model,weight),block) in enumerate(three.groupby(["model","weighting"],sort=True)):
        ordered=block.set_index("regime").loc[list(REGIMES)]
        ax.bar(x+(k-1.5)*width,ordered["mean_total_turnover"],width,label=f"{model} {weight}")
    ax.set_xticks(x,list(REGIMES));ax.set_ylabel("One-sided D10 + D1 turnover");ax.legend()
    fig.tight_layout();fig.savefig(output_dir/"turnover_by_regime.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    part=net[net["transaction_cost_bps"]==50]
    for (model,weight),block in part.groupby(["model","weighting"],sort=True):
        block=block.sort_values("formation_date")
        ax.plot(block["formation_date"],block["D10_minus_D1_net"].cumsum(),label=f"{model} {weight}")
    ax.axhline(0,color="black",linewidth=.8)
    ax.set_ylabel("Cumulative arithmetic net D10-D1 (not compounded)")
    ax.legend();fig.tight_layout();fig.savefig(output_dir/"net_cumulative_long_short_50bps.png",dpi=180);plt.close(fig)


def _latex_table(table: pd.DataFrame) -> str:
    lines=[r"\begin{table}[htbp]",r"\centering",r"\caption{Monthly gross versus 50-bp net $D10-D1$ by VIX regime}",r"\label{tab:transaction-costs}",r"\small",r"\begin{tabular}{lllrrrrrrr}",r"\hline",r"Model & Weight & Regime & Turnover & Gross & Net & HAC SE & $t$ & $p$ & $N$ \\",r"\hline"]
    for row in table.itertuples(index=False):
        lines.append(f"{row.model} & {row.weighting} & {row.regime} & {row.mean_total_turnover:.3f} & {row.mean_gross_D10_minus_D1:.4f} & {row.mean_net_D10_minus_D1:.4f} & {row.hac_se:.4f} & {row.t_stat:.2f} & {row.p_value:.4f} & {int(row.n_months)} " + r"\\")
    lines.extend([r"\hline",r"\end{tabular}",r"\begin{flushleft}",r"\small Notes: 50 basis points per one-sided turnover dollar; turnover is $D10+D1$ rebalancing turnover (initial leg turnover=1). Drift uses prior holding-period returns realized by each formation date. HAC uses six monthly lags. LOW/MIDDLE/HIGH use expanding formation-time VIX terciles.",r"\end{flushleft}",r"\end{table}",""])
    return "\n".join(lines)


def _summary_markdown(summary: pd.DataFrame, inf: pd.DataFrame, diff: pd.DataFrame, breakeven: pd.DataFrame) -> str:
    lines=["# Turnover and Transaction-Cost Analysis", "", "## Methodology",
           "Authoritative M8 assignments are reused unchanged (tie broken deterministically by PERMNO). EW is 1/N and VW uses positive, finite lagged formation ME. Previous targets drift through the prior realized holding-period returns before rebalance.",
           "One-sided turnover equals 0.5 times the L1 target-minus-pretrade distance; each newly cash-funded leg incurs initial turnover 1. The D10-D1 spread is per $1 long and $1 short and incurs costs on both legs: 0.005 × (turnover D10 + turnover D1).",
           "The D1_net field is the **long-only** D1 net return; D1_short_leg_net_contribution is the actual short-side contribution to D10-D1 net. Therefore D10_net - D1_net is **not** the net long-short spread.",
           "Gross and net cumulative lines use M8 cumulative arithmetic returns, not compounded wealth.",
           "", "## 50-basis-point overall and LOW/MIDDLE/HIGH results"]
    view=summary[(summary.transaction_cost_bps==50)&(summary.regime_definition.isin(["overall","three_state"]))]
    for row in view.itertuples(index=False):
        inf_row=inf[(inf.model==row.model)&(inf.weighting==row.weighting)&(inf.regime==row.regime)].iloc[0]
        lines.append(f"- {row.model} {row.weighting} {row.regime}: N={row.n_months}, turnover={row.mean_total_turnover:.3f}, gross={row.mean_gross_D10_minus_D1:.4%}, net={row.mean_net_D10_minus_D1:.4%}, net HAC t={inf_row.t_stat:.2f}, p={inf_row.p_value:.4f}.")
    lines.extend(["", "## Three-state net regime differences (Holm primary, BH FDR secondary)"])
    for row in diff.itertuples(index=False):
        lines.append(f"- {row.model} {row.weighting} {row.comparison}: difference={row.estimated_difference:.5f}, raw p={row.raw_p_value:.4f}, Holm p={row.holm_p_value:.4f}.")
    lines.extend(["", "## Break-even one-way costs (descriptive)"])
    for row in breakeven[breakeven.regime_definition=="overall"].itertuples(index=False):
        lines.append(f"- {row.model} {row.weighting}: {row.breakeven_cost_bps:.2f} bps" if np.isfinite(row.breakeven_cost_bps) else f"- {row.model} {row.weighting}: undefined")
    lines.extend(["", "## Limitations", "50 bps is a stylized proportional one-way trading-cost robustness assumption, **not** a measured execution cost. Spread, liquidity, nonlinear market impact, trade size and period can all affect realizable implementation costs. Short borrow/locate fees, rebates, financing, and nonlinear market impact are not separately modeled. Positive modeled net returns do not establish an implementable trading strategy.",""])
    return "\n".join(lines)


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_transaction_costs(root: Path) -> dict[str, Path]:
    """Fail closed before writing M10 outputs if any M8 or M9 invariant changes."""
    root=Path(root).resolve()
    economic=root/"results"/"economic_value"
    inference=root/"results"/"inference"
    files={
        "assignments":economic/"portfolio_assignments.csv",
        "monthly_gross":economic/"decile_returns_monthly.csv",
        "long_short_gross":economic/"long_short_returns.csv",
        "M9_inference":inference/"long_short_hac_inference.csv",
        "M9_differences":inference/"long_short_regime_differences.csv",
        "predictions":root/"results"/"baseline_models"/"baseline_oos_predictions.parquet",
    }
    for key,path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Frozen {key} input is unavailable: {path}. Restore existing artifacts; do not retrain")
    original_hashes={key:_sha(path) for key,path in files.items()}
    print("[1/8] Loading frozen M8 portfolio assignments and gross returns...")
    assignments=pd.read_csv(files["assignments"])
    gross=pd.read_csv(files["monthly_gross"])
    ls=pd.read_csv(files["long_short_gross"])
    print("[2/8] Reconstructing M8 EW/VW target and drifted pretrade weights...")
    rebuilt, turnover, weights=reconstruct_portfolios(assignments)
    print("[3/8] Computing decile and long-short turnover...")
    print("[4/8] Validating exact zero-cost M8 gross replication and frozen M9 HAC inference...")
    validate_gross_replication(rebuilt,gross)
    validate_authoritative_long_short(rebuilt,ls)
    m9_recalc=long_short_hac_inference(_normalize_dates(ls))
    m9_diff=long_short_regime_differences(_normalize_dates(ls))
    _compare_gross_m9(m9_recalc,pd.read_csv(files["M9_inference"]),
                      ["model","weighting","regime_definition","regime"],
                      ["mean_monthly_return","hac_se","t_stat","p_value","ci_lower","ci_upper","n_months","hac_lag"])
    _compare_gross_m9(m9_diff,pd.read_csv(files["M9_differences"]),
                      ["model","weighting","comparison"],
                      ["estimated_difference","hac_se","t_stat","raw_p_value","holm_p_value","bh_fdr_p_value","ci_lower","ci_upper","n_months","hac_lag"])
    print("[5/8] Applying frozen 0 and 50 bps one-way proportional costs...")
    net=apply_cost_scenarios(rebuilt)
    zero=net[net.transaction_cost_bps==0]
    if not np.array_equal(zero.D10_minus_D1_net.to_numpy(),zero.D10_minus_D1_gross.to_numpy()):
        raise AssertionError("0-bps gross replication failed")
    long_turn=rebuilt[["formation_date","model","weighting","D10_turnover","D1_turnover","regime_binary","regime_3state"]].copy()
    long_turn["long_short_total_turnover"]=long_turn.D10_turnover+long_turn.D1_turnover
    turnover_summary=_turnover_summary(rebuilt,turnover)
    summary, breakeven=_transaction_summaries(net)
    print("[6/8] Reusing M9 six-lag HAC mean and regime-difference methods on 50-bps net spread...")
    net_50=net[net.transaction_cost_bps==50].copy()
    # Preserve exactly the M9 interface with no alternative statistical estimator.
    inference_input=net_50.rename(columns={"D10_minus_D1_net":"D10_minus_D1"})
    net_inf=long_short_hac_inference(inference_input).rename(columns={"mean_monthly_return":"mean_net_return"})
    net_inf.insert(2,"transaction_cost_bps",50)
    net_diff=long_short_regime_differences(inference_input)
    net_diff.insert(2,"transaction_cost_bps",50)
    print("[7/8] Generating state-conditioned summaries, break-even diagnostics and publication table...")
    publication=summary[(summary.transaction_cost_bps==50) & (summary.regime_definition.isin(["overall","three_state"]))].merge(
        net_inf[["model","weighting","regime","hac_se","t_stat","p_value"]],
        on=["model","weighting","regime"],validate="one_to_one")
    publication=publication.sort_values(["model","weighting","regime_definition","regime"])
    if len(publication)!=16:
        raise AssertionError("Expected 16 primary publication rows: 4 model/weight series × 4 regimes")
    # All protected inputs must be byte-identical even if the run fails later.
    if {key:_sha(path) for key,path in files.items()}!=original_hashes:
        raise AssertionError("A protected M8/M9 input changed during M10 execution")
    print("[8/8] Writing new isolated transaction-cost research artifacts...")
    out=root/"results"/"transaction_costs"
    out.mkdir(parents=True,exist_ok=True)
    outputs={
        "portfolio_weights_monthly.csv": weights,
        "turnover_monthly.csv":turnover,
        "long_short_turnover.csv":long_turn,
        "net_portfolio_returns.csv":net,
        "transaction_cost_summary.csv":summary,
        "turnover_summary.csv":turnover_summary,
        "breakeven_costs.csv":breakeven,
        "net_long_short_hac_inference.csv":net_inf,
        "net_regime_differences.csv":net_diff,
        "publication_transaction_cost_table.csv":publication,
    }
    for name,frame in outputs.items():
        frame.to_csv(out/name,index=False,float_format="%.17g")
    (out/"transaction_cost_summary.md").write_text(_summary_markdown(summary,net_inf,net_diff,breakeven),encoding="utf-8")
    _plot_summaries(summary,net,out)
    paper_dir=root/"paper"/"generated"
    paper_dir.mkdir(parents=True,exist_ok=True)
    (paper_dir/"table_transaction_costs.tex").write_text(_latex_table(publication),encoding="utf-8")
    metadata={
        "transaction_cost_bps":list(PRIMARY_BPS),"basis_point_definition":"bps / 10000",
        "one_way_turnover":"initial=1; otherwise 0.5*sum(abs(target-pretrade))",
        "drift":"previous formation targets multiplied by 1+previous realized t return, normalized at current t",
        "first_month":"charge one unit turnover on each cash-financed D1/D10 leg",
        "long_short_normalization":"$1 D10 long and $1 D1 short; total gross $2; costs .005*(turnover D10+turnover D1)",
        "weighting":{"EW":"1/N valid assigned stocks","VW":"positive finite M8 weighting_variable / eligible decile sum"},
        "hac_lag":BASELINE_HAC_LAG,"inference":"M9 statsmodels HAC intercept and LOW-reference regression/contrasts; Holm primary, BH FDR secondary",
        "regimes":{"binary":"formation-time expanding VIX median","three_state":"formation-time expanding VIX terciles"},
        "source_hashes_sha256":original_hashes,"source_paths":{k:str(v.relative_to(root)) for k,v in files.items()},
        "git_commit_at_execution":_git_commit(root),"python":platform.python_version(),
        "numpy":np.__version__,"pandas":pd.__version__,"matplotlib":matplotlib.__version__,
        "gross_M8_reproduced":True,"gross_M9_inference_reproduced":True,
    }
    (out/"run_metadata.json").write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if {key:_sha(path) for key,path in files.items()}!=original_hashes:
        raise AssertionError("A protected M8/M9 input changed during M10 output writing")
    return {"output_dir":out,"summary":out/"transaction_cost_summary.md",
            "publication_table":paper_dir/"table_transaction_costs.tex"}

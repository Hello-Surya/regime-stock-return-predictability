"""Leakage-aware monthly feature construction."""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_CRSP_COLUMNS = {"permno", "date", "ret", "prc", "shrout"}


def _monthly_series(group: pd.DataFrame, column: str) -> pd.Series:
    indexed = group.set_index("date")[column].sort_index()
    full_index = pd.date_range(indexed.index.min(), indexed.index.max(), freq="ME")
    return indexed.reindex(full_index)


def _calendar_safe_features(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("date").copy()
    ret = _monthly_series(g, "ret")
    me = _monthly_series(g, "market_equity")
    momentum = ret.shift(2).rolling(11, min_periods=11).apply(
        lambda x: np.prod(1.0 + x) - 1.0, raw=True
    )
    next_return = ret.shift(-1)
    me_lag = me.shift(1)
    g["mom_12_2"] = g["date"].map(momentum)
    g["next_month_return"] = g["date"].map(next_return)
    g["me_lag"] = g["date"].map(me_lag)
    g["realized_return_date"] = g["date"] + pd.offsets.MonthEnd(1)
    g["me_lag_date"] = g["date"] - pd.offsets.MonthEnd(1)
    return g


def build_features(crsp: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_CRSP_COLUMNS.difference(crsp.columns)
    if missing:
        raise ValueError(f"CRSP input is missing required columns: {sorted(missing)}")
    df = crsp.copy()
    df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
    if df.duplicated(["permno", "date"]).any():
        raise ValueError("CRSP input contains duplicate PERMNO-date observations.")
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)
    df["prc"] = pd.to_numeric(df["prc"], errors="coerce").abs()
    df["shrout"] = pd.to_numeric(df["shrout"], errors="coerce")
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    df["market_equity"] = df["prc"] * df["shrout"]
    df["log_me"] = np.log(df["market_equity"].where(df["market_equity"] > 0))
    pieces = [_calendar_safe_features(group) for _, group in df.groupby("permno", sort=False)]
    out = pd.concat(pieces, ignore_index=True) if pieces else df.copy()
    if "book_equity" in out.columns:
        out["book_to_market"] = out["book_equity"] / out["market_equity"].replace(0, np.nan)
    return out.sort_values(["permno", "date"]).reset_index(drop=True)

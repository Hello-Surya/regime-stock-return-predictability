"""VIX-based high/low regime labeling using only information available through t."""
from __future__ import annotations

import pandas as pd


def label_vix_regime(vix_monthly: pd.DataFrame, date_col: str = "date", vix_col: str = "vix") -> pd.DataFrame:
    if vix_monthly.empty:
        return pd.DataFrame(columns=[date_col, vix_col, "vix_expanding_median", "regime"])
    if date_col not in vix_monthly or vix_col not in vix_monthly:
        raise ValueError(f"VIX data must contain '{date_col}' and '{vix_col}'.")
    out = vix_monthly[[date_col, vix_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col).drop_duplicates(date_col, keep="last").reset_index(drop=True)
    out["vix_expanding_median"] = out[vix_col].expanding(min_periods=1).median()
    out["regime"] = (out[vix_col] > out["vix_expanding_median"]).map({True: "HIGH", False: "LOW"})
    return out

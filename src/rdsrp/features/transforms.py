"""Cross-sectional transforms that do not use future months."""
from __future__ import annotations

import numpy as np
import pandas as pd


def standardize_by_month(df: pd.DataFrame, cols: list[str], date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        grouped = out.groupby(date_col)[col]
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        out[f"{col}_z"] = (out[col] - mean) / std
    return out


def winsorize_by_month(df: pd.DataFrame, cols: list[str], lower: float = 0.01, upper: float = 0.99, date_col: str = "date") -> pd.DataFrame:
    if not 0 <= lower < upper <= 1:
        raise ValueError("Winsorization quantiles must satisfy 0 <= lower < upper <= 1.")
    out = df.copy()
    for col in cols:
        low = out.groupby(date_col)[col].transform(lambda x: x.quantile(lower))
        high = out.groupby(date_col)[col].transform(lambda x: x.quantile(upper))
        out[col] = out[col].clip(low, high)
    return out

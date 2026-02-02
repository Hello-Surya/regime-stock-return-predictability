"""VIX-based two-regime labeling (High vs Low) using rolling median.

Important: rolling median at month t uses values up to and including t only.
"""
from __future__ import annotations

import pandas as pd


def label_vix_regime(vix_monthly: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Return a DataFrame with columns: date, vix, rolling_median, regime."""
    # Stub
    return pd.DataFrame()

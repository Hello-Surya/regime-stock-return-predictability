"""Transaction cost adjustments: baseline 0 bps, robustness 50 bps one-way."""
from __future__ import annotations

import pandas as pd


def apply_one_way_costs(ls_returns: pd.Series, turnover: pd.Series, bps: float) -> pd.Series:
    """Apply one-way costs in bps using turnover."""
    # Stub
    return ls_returns

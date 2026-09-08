"""Date helpers."""
from __future__ import annotations

import pandas as pd


def to_month_end(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values) + pd.offsets.MonthEnd(0)

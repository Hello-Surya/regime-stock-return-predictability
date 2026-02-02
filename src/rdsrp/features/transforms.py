"""Transforms: winsorization, standardization, missing-value handling (time-safe)."""
from __future__ import annotations

import pandas as pd


def standardize_by_month(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cross-sectional standardization within each month."""
    # Stub
    return df

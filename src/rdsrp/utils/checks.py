"""Assertions and invariant checks used throughout the pipeline."""
from __future__ import annotations

import pandas as pd


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    """Raise ValueError if any required columns are missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

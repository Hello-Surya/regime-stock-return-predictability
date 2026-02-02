"""VIX data source and monthly alignment.

Regime definition is VIX rolling median using only info up to month t.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_vix_series(path: Path) -> pd.DataFrame:
    """Load a VIX time series and return monthly-aligned values."""
    # Stub
    return pd.DataFrame()

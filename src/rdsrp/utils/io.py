"""Lightweight IO helpers (parquet/csv) with consistent options."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write parquet with stable settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    """Read parquet."""
    return pd.read_parquet(path)

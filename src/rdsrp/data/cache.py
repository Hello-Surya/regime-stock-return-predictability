"""Parquet caching helpers for local research data."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def write_parquet_cache(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_parquet_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)

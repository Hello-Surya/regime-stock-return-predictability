"""WRDS data access for CRSP monthly US common stocks (shrcd 10/11).

This module must never commit raw CRSP outputs to git.
Provide functions that pull and store locally under data/raw/ (ignored).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class WrdsCredentials:
    """WRDS credentials holder (do not hardcode; use env vars)."""
    username: str


def pull_crsp_monthly(
    out_path: Path,
    start: str,
    end: str,
    username: Optional[str] = None,
) -> Path:
    """Pull CRSP monthly stock data from WRDS and save locally (parquet).

    Expected filters:
    - exchanges: NYSE/AMEX/NASDAQ
    - shrcd: 10/11
    - frequency: monthly
    """
    # Stub: implement with wrds or sqlalchemy connection
    # Return the path where data is saved.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path

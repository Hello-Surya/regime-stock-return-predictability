"""Decile portfolio construction and 10-1 long-short returns (EW and VW)."""
from __future__ import annotations

import pandas as pd


def decile_long_short(
    preds: pd.DataFrame,
    ret_col: str = "ret_fwd",
    pred_col: str = "pred",
    weight_col: str | None = None,
) -> pd.DataFrame:
    """Compute monthly decile returns and 10-1 long-short series."""
    # Stub
    return pd.DataFrame()

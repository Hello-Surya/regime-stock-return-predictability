"""Feature construction for monthly CRSP panel.

Builds X(t) using only information available at time t.
Target is next-month return r(t+1).
"""
from __future__ import annotations

import pandas as pd


def build_features(crsp: pd.DataFrame) -> pd.DataFrame:
    """Return a modeling panel with predictors at t and target at t+1."""
    # Stub
    return pd.DataFrame()

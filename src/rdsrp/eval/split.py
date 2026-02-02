"""Expanding-window out-of-sample split logic."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class SplitSpec:
    """Defines expanding-window OOS schedule."""
    start_train: str
    start_test: str
    end: str


def expanding_window_splits(dates: pd.Series, spec: SplitSpec):
    """Yield (train_mask, test_mask) pairs by month."""
    # Stub (generator)
    yield from ()

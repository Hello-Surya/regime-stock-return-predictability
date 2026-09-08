"""Expanding-window out-of-sample split logic."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator
import pandas as pd

@dataclass(frozen=True)
class SplitSpec:
    start_train: str
    start_test: str
    end: str


def expanding_window_splits(dates: pd.Series, spec: SplitSpec) -> Iterator[tuple[pd.Series, pd.Series]]:
    d = pd.to_datetime(dates)
    start_train = pd.Timestamp(spec.start_train)
    start_test = pd.Timestamp(spec.start_test)
    end = pd.Timestamp(spec.end)
    test_months = sorted(x for x in d.dropna().unique() if start_test <= x <= end)
    for test_date in test_months:
        train_mask = (d >= start_train) & (d < test_date)
        test_mask = d == test_date
        if train_mask.any() and test_mask.any():
            yield train_mask, test_mask

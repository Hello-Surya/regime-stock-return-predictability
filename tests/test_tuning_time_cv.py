import pandas as pd

from rdsrp.models.tuning import forward_chaining_splits


def test_forward_chaining_tuning_never_uses_future_dates():
    dates = pd.Series(pd.date_range("2010-01-31", periods=24, freq="ME").repeat(4))
    splits = list(forward_chaining_splits(dates, n_splits=3, min_train_months=12))
    assert len(splits) == 3
    for train_idx, valid_idx in splits:
        assert dates.iloc[train_idx].max() < dates.iloc[valid_idx].min()

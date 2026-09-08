import pandas as pd

from rdsrp.eval.split import SplitSpec, expanding_window_splits


def test_expanding_window_train_precedes_test() -> None:
    dates = pd.Series(pd.date_range("2000-01-31", periods=4, freq="ME"))
    spec = SplitSpec(start_train="2000-01-31", start_test="2000-02-29", end="2000-04-30")
    splits = list(expanding_window_splits(dates, spec))
    assert len(splits) == 3
    for train, test in splits:
        assert dates[train].max() < dates[test].min()

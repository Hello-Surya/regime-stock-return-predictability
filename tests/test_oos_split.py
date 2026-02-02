def test_oos_split():
    import pandas as pd
    from rdsrp.eval.split import SplitSpec, expanding_window_splits

    dates = pd.Series(pd.date_range('2000-01-31', periods=3, freq='M'))
    spec = SplitSpec(start_train='2000-01-31', start_test='2000-02-29', end='2000-03-31')
    splits = list(expanding_window_splits(dates, spec))
    assert splits == []

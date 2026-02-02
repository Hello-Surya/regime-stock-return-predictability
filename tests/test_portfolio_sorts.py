def test_portfolio_sorts():
    import pandas as pd
    from rdsrp.portfolios.sorts import decile_long_short

    preds = pd.DataFrame()
    res = decile_long_short(preds)
    assert res.empty

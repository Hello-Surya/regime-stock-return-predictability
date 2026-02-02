def test_regime_labeling():
    import pandas as pd
    from rdsrp.regimes.vix_regime import label_vix_regime

    # A simple test with empty DataFrame should return empty DataFrame
    df = pd.DataFrame()
    out = label_vix_regime(df)
    assert out.empty

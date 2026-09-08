import pandas as pd

from rdsrp.regimes.vix_regime import label_vix_regime


def test_empty_regime_input() -> None:
    assert label_vix_regime(pd.DataFrame()).empty


def test_expanding_median_uses_no_future_vix() -> None:
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-31", periods=4, freq="ME"),
        "vix": [10.0, 30.0, 20.0, 100.0],
    })
    out = label_vix_regime(df)
    assert out["vix_expanding_median"].tolist() == [10.0, 20.0, 20.0, 25.0]
    assert out["regime"].tolist() == ["LOW", "HIGH", "LOW", "HIGH"]

    changed_future = df.copy()
    changed_future.loc[3, "vix"] = 1000.0
    earlier = label_vix_regime(changed_future)
    pd.testing.assert_series_equal(
        out.loc[:2, "vix_expanding_median"],
        earlier.loc[:2, "vix_expanding_median"],
    )

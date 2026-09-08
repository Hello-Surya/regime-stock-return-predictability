import numpy as np
import pandas as pd

from rdsrp.features.build import build_features


def _panel(returns: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2000-01-31", periods=len(returns), freq="ME")
    return pd.DataFrame({
        "permno": 1,
        "ticker": "TEST",
        "date": dates,
        "ret": returns,
        "prc": 10.0,
        "shrout": 100.0,
    })


def test_next_month_return_alignment() -> None:
    df = build_features(_panel([0.01, 0.02, -0.03, 0.04]))
    assert df.loc[0, "next_month_return"] == 0.02
    assert df.loc[1, "next_month_return"] == -0.03
    assert np.isnan(df.loc[3, "next_month_return"])


def test_momentum_12_2_excludes_t_and_t_minus_1() -> None:
    returns = [0.01] * 11 + [0.50, 0.90]
    df = build_features(_panel(returns))
    expected = (1.01**11) - 1.0
    assert np.isclose(df.loc[12, "mom_12_2"], expected)


def test_calendar_gap_is_not_treated_as_next_month() -> None:
    df = pd.DataFrame({
        "permno": [1, 1, 1],
        "ticker": ["TEST"] * 3,
        "date": pd.to_datetime(["2020-01-31", "2020-03-31", "2020-04-30"]),
        "ret": [0.01, 0.03, 0.04],
        "prc": [10.0, 10.3, 10.7],
        "shrout": [100.0, 100.0, 100.0],
    })
    out = build_features(df)
    jan = out[out["date"] == pd.Timestamp("2020-01-31")].iloc[0]
    mar = out[out["date"] == pd.Timestamp("2020-03-31")].iloc[0]
    assert pd.isna(jan["next_month_return"])
    assert pd.isna(mar["me_lag"])
    assert mar["next_month_return"] == 0.04


def test_feature_month_price_screen_does_not_remove_future_target() -> None:
    raw = pd.DataFrame({
        "permno": [1, 1],
        "ticker": ["TEST", "TEST"],
        "date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
        "ret": [0.01, -0.40],
        "prc": [10.0, 4.0],
        "shrout": [100.0, 100.0],
    })
    features = build_features(raw)
    screened = features[features["prc"] >= 5.0]
    assert len(screened) == 1
    assert screened.iloc[0]["next_month_return"] == -0.40

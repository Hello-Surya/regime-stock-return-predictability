import pandas as pd

from rdsrp.portfolios.sorts import assign_deciles


def test_prediction_deciles_are_ordered() -> None:
    df = pd.DataFrame({
        "date": pd.Timestamp("2020-01-31"),
        "prediction": list(range(20)),
    })
    out = assign_deciles(df)
    assert out.loc[out["prediction"].idxmin(), "decile"] == 1
    assert out.loc[out["prediction"].idxmax(), "decile"] == 10

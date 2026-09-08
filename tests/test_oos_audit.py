import numpy as np
import pandas as pd

from rdsrp.eval.oos import run_expanding_oos
from rdsrp.models.base import FitResult, ModelSpec


class MeanModel(ModelSpec):
    name = "mean"

    def fit(self, X, y):
        class Estimator:
            def __init__(self, value):
                self.value = value

            def predict(self, values):
                return np.repeat(self.value, len(values))

        return FitResult(
            model_name=self.name,
            estimator=Estimator(float(np.mean(y))),
            params={},
        )


def test_oos_exposes_formation_target_training_and_regime_audits():
    dates = pd.date_range("2019-01-31", periods=16, freq="ME")
    rows = []
    for date in dates:
        for permno in range(1, 11):
            rows.append(
                {
                    "date": date,
                    "permno": permno,
                    "ticker": f"S{permno}",
                    "log_me": float(permno),
                    "mom_12_2": float(permno) / 10,
                    "next_month_return": 0.01 * permno,
                    "realized_return_date": date + pd.offsets.MonthEnd(1),
                    "regime": "HIGH" if date.month % 2 else "LOW",
                    "vix": 20.0,
                    "me_lag": 100.0 + permno,
                    "me_lag_date": date - pd.offsets.MonthEnd(1),
                }
            )
    panel = pd.DataFrame(rows)
    out = run_expanding_oos(
        panel,
        feature_cols=["log_me", "mom_12_2"],
        models={"mean": MeanModel()},
        start_test=dates[-2],
        min_train_months=12,
    )
    assert not out.empty
    assert (out["formation_date"] == out["date"]).all()
    assert (out["realized_return_date"] > out["formation_date"]).all()
    assert (out["train_feature_end_date"] < out["formation_date"]).all()
    assert (out["train_target_end_date"] <= out["formation_date"]).all()
    assert (out["value_weight_date"] <= out["formation_date"]).all()

    expected_regime = panel[["date", "regime"]].drop_duplicates().set_index("date")["regime"]
    assert out["regime"].tolist() == out["formation_date"].map(expected_regime).tolist()

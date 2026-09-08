import warnings

import numpy as np
import pandas as pd

from rdsrp.portfolios.strict_evaluation import (
    monthly_rank_metrics_strict,
    prediction_group_diagnostics,
    prepare_portfolio_assignments_strict,
)


def _panel(predictions: list[float], model: str = "elastic_net") -> pd.DataFrame:
    formation = pd.Timestamp("2020-01-31")
    realized = pd.Timestamp("2020-02-29")
    n = len(predictions)
    return pd.DataFrame(
        {
            "formation_date": [formation] * n,
            "realized_return_date": [realized] * n,
            "permno": np.arange(10000, 10000 + n),
            "ticker": [f"S{i:03d}" for i in range(n)],
            "model": [model] * n,
            "prediction": predictions,
            "actual_next_month_return": np.linspace(-0.05, 0.05, n),
            "regime": ["HIGH"] * n,
            "me_lag": np.arange(1.0, n + 1.0),
            "value_weight_date": [pd.Timestamp("2019-12-31")] * n,
            "train_feature_end_date": [pd.Timestamp("2019-12-31")] * n,
            "train_target_end_date": [formation] * n,
        }
    )


def test_constant_predictions_are_not_sorted_into_deciles():
    frame = _panel([0.01] * 20)
    assignments, diagnostics = prepare_portfolio_assignments_strict(frame)
    assert assignments["decile"].isna().all()
    assert set(assignments["decile_status"]) == {"fewer_than_10_distinct_predictions"}
    assert len(diagnostics.skipped_months) == 1
    row = diagnostics.skipped_months.iloc[0]
    assert row["n_unique_predictions"] == 1
    assert row["reason"] == "fewer_than_10_distinct_predictions"


def test_partial_ties_remain_sortable_when_ten_distinct_forecasts_exist():
    predictions = [float(i // 2) for i in range(20)]
    assignments, diagnostics = prepare_portfolio_assignments_strict(_panel(predictions))
    assert diagnostics.skipped_months.empty
    assert set(assignments["decile"].dropna().astype(int)) == set(range(1, 11))


def test_near_constant_numerical_noise_is_rejected():
    predictions = [0.01 + i * 1e-14 for i in range(20)]
    diagnostics = prediction_group_diagnostics(_panel(predictions))
    assert diagnostics.iloc[0]["prediction_status"] == "near_constant_predictions"


def test_constant_predictions_produce_nan_rank_ic_without_warning():
    frame = _panel([0.01] * 20)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        monthly = monthly_rank_metrics_strict(frame)
    assert pd.isna(monthly.iloc[0]["spearman"])
    assert monthly.iloc[0]["n_unique_predictions"] == 1
    assert not caught


def test_nonconstant_predictions_still_compute_monthly_rank_ic():
    frame = _panel(list(np.linspace(-0.1, 0.1, 20)), model="xgboost")
    monthly = monthly_rank_metrics_strict(frame)
    assert np.isclose(monthly.iloc[0]["spearman"], 1.0)
    assert monthly.iloc[0]["n_unique_predictions"] == 20

import numpy as np
import pandas as pd

from rdsrp.baseline.economic_value import (
    assign_prediction_deciles,
    hac_mean_inference,
    hac_regime_difference,
    monthly_decile_long_short_returns,
    one_way_turnover,
    portfolio_regime_difference_tests,
    rank_ic_inference,
    summarize_portfolio_performance,
)


def _prediction_panel(n_months=12, n_stocks=100):
    rows = []
    for i, date in enumerate(pd.date_range("2020-01-31", periods=n_months, freq="ME")):
        regime = "LOW" if i % 2 == 0 else "HIGH"
        for permno in range(1, n_stocks + 1):
            score = (permno - 50.0) / 50.0
            realized = 0.01 * score if regime == "LOW" else 0.0
            rows.append(
                {
                    "date": date,
                    "permno": permno,
                    "regime": regime,
                    "actual_next_month_return": realized,
                    "me_lag": float(100 + permno),
                    "me_lag_date": date + pd.offsets.MonthEnd(-1),
                    "elastic_net_prediction": score,
                    "xgboost_prediction": score,
                }
            )
    return pd.DataFrame(rows)


def test_tie_preserving_deciles_do_not_break_equal_predictions_by_row_order():
    values = pd.Series([0.0] * 10 + [1.0] * 10)
    deciles = assign_prediction_deciles(values)
    assert deciles.iloc[:10].nunique() == 1
    assert deciles.iloc[10:].nunique() == 1
    assert deciles.iloc[0] != deciles.iloc[-1]


def test_turnover_zero_for_unchanged_weights_and_one_for_initial_side():
    weights = {1: 0.5, 2: 0.5}
    assert one_way_turnover(weights, None) == 1.0
    state = (weights, {1: 0.0, 2: 0.0})
    assert np.isclose(one_way_turnover(weights, state), 0.0)


def test_value_weight_timing_is_enforced():
    panel = _prediction_panel(n_months=3, n_stocks=20)
    panel.loc[0, "me_lag_date"] = panel.loc[0, "date"]
    try:
        monthly_decile_long_short_returns(panel)
    except AssertionError as exc:
        assert "previous calendar month-end" in str(exc)
    else:
        raise AssertionError("Expected the lagged-market-equity timing audit to fail.")


def test_monthly_portfolios_apply_costs_and_low_regime_has_signal():
    panel = _prediction_panel()
    monthly = monthly_decile_long_short_returns(panel, transaction_cost_bps=50.0)
    assert set(monthly["weighting"]) == {"EW", "VW"}
    assert set(monthly["model"]) == {"Elastic Net", "XGBoost"}
    assert (monthly["transaction_cost"] >= 0).all()
    assert np.allclose(
        monthly["net_long_short_return"],
        monthly["gross_long_short_return"] - monthly["transaction_cost"],
    )
    low = monthly[(monthly.model == "XGBoost") & (monthly.weighting == "EW") & (monthly.regime == "LOW")]
    high = monthly[(monthly.model == "XGBoost") & (monthly.weighting == "EW") & (monthly.regime == "HIGH")]
    assert low["gross_long_short_return"].mean() > high["gross_long_short_return"].mean()


def test_hac_mean_and_regime_difference_have_expected_signs():
    values = pd.Series([0.00, 0.02, 0.00, 0.02, 0.00, 0.02, 0.00, 0.02])
    regimes = pd.Series(["HIGH", "LOW"] * 4)
    mean = hac_mean_inference(values, maxlags=1)
    diff = hac_regime_difference(values, regimes, maxlags=1)
    assert np.isclose(mean["mean"], 0.01)
    assert np.isclose(diff["high_mean"], 0.0)
    assert np.isclose(diff["low_mean"], 0.02)
    assert diff["low_minus_high"] > 0


def test_portfolio_and_rank_inference_tables_cover_required_groups():
    panel = _prediction_panel()
    monthly = monthly_decile_long_short_returns(panel)
    perf = summarize_portfolio_performance(monthly)
    tests = portfolio_regime_difference_tests(monthly)
    assert len(perf) == 12
    assert len(tests) == 8
    assert {"OVERALL", "HIGH", "LOW"} == set(perf["regime"])

    rank = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=12, freq="ME").repeat(2),
            "regime": np.repeat(["LOW", "HIGH"] * 6, 2),
            "model": ["Elastic Net", "XGBoost"] * 12,
            "spearman_ic": np.tile([0.01, 0.02], 12),
        }
    )
    within, diff = rank_ic_inference(rank)
    assert len(within) == 6
    assert len(diff) == 2

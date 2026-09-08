import numpy as np
import pandas as pd

from rdsrp.portfolios.evaluation import (
    assign_deciles,
    compute_decile_returns,
    fixed_universe_permnos,
    long_short_returns,
    monthly_rank_metrics,
    prepare_portfolio_assignments,
    summarize_portfolios,
    validate_portfolio_outputs,
)


def _prediction_panel(n=20, models=("elastic_net",), date="2020-01-31"):
    rows = []
    formation = pd.Timestamp(date)
    realized = formation + pd.offsets.MonthEnd(1)
    for model in models:
        for i in range(n):
            rows.append(
                {
                    "formation_date": formation,
                    "realized_return_date": realized,
                    "permno": 10000 + i,
                    "ticker": f"S{i:03d}",
                    "model": model,
                    "prediction": float(i),
                    "actual_next_month_return": float(i) / 100.0,
                    "regime": "HIGH",
                    "me_lag": float(i + 1),
                    "value_weight_date": formation - pd.offsets.MonthEnd(1),
                    "train_feature_end_date": formation - pd.offsets.MonthEnd(1),
                    "train_target_end_date": formation,
                }
            )
    return pd.DataFrame(rows)


def test_decile_direction_and_tie_handling_are_deterministic():
    df = _prediction_panel(20)
    df.loc[:9, "prediction"] = 1.0
    first = assign_deciles(df, date_col="formation_date")
    second = assign_deciles(df.sample(frac=1, random_state=4), date_col="formation_date").sort_values("permno")
    first = first.sort_values("permno")
    assert first.iloc[0]["decile"] == 1
    assert first.loc[first["prediction"].idxmax(), "decile"] == 10
    assert first["decile"].tolist() == second["decile"].tolist()


def test_cross_sectional_isolation_by_month_and_model():
    a = _prediction_panel(20, models=("elastic_net", "xgboost"), date="2020-01-31")
    b = _prediction_panel(20, models=("elastic_net", "xgboost"), date="2020-02-29")
    b["prediction"] = -b["prediction"]
    out = assign_deciles(pd.concat([a, b], ignore_index=True), date_col="formation_date")
    for _, block in out.groupby(["formation_date", "model"]):
        assert set(block["decile"].dropna().astype(int)) == set(range(1, 11))
        low = block.loc[block["prediction"].idxmin(), "decile"]
        high = block.loc[block["prediction"].idxmax(), "decile"]
        assert low == 1
        assert high == 10


def test_insufficient_securities_are_not_called_deciles():
    out = assign_deciles(_prediction_panel(9), date_col="formation_date")
    assert out["decile"].isna().all()
    assert set(out["decile_status"]) == {"insufficient_securities"}


def test_equal_weight_value_weight_and_long_short_arithmetic():
    df = _prediction_panel(20)
    assignments, _ = prepare_portfolio_assignments(df)
    returns = compute_decile_returns(assignments)
    ew = returns[returns["weighting"] == "EW"].iloc[0]
    vw = returns[returns["weighting"] == "VW"].iloc[0]
    d1 = assignments[assignments["decile"] == 1]
    expected_ew = d1["actual_next_month_return"].mean()
    expected_vw = np.average(d1["actual_next_month_return"], weights=d1["weighting_variable"])
    assert np.isclose(ew["D1"], expected_ew)
    assert np.isclose(vw["D1"], expected_vw)
    assert np.isclose(ew["D10_minus_D1"], ew["D10"] - ew["D1"])


def test_invalid_weights_do_not_fall_back_to_equal_weighting():
    df = _prediction_panel(30)
    df.loc[:2, "me_lag"] = [np.nan, -1.0, np.inf]
    assignments, _ = prepare_portfolio_assignments(df)
    returns = compute_decile_returns(assignments)
    vw = returns[returns["weighting"] == "VW"].iloc[0]
    assert pd.isna(vw["D1"])
    assert vw["D1_n_weight"] == 0


def test_rank_ic_is_monthly_not_pooled():
    a = _prediction_panel(20, date="2020-01-31")
    b = _prediction_panel(20, date="2020-02-29")
    b["actual_next_month_return"] = -b["actual_next_month_return"]
    monthly = monthly_rank_metrics(pd.concat([a, b], ignore_index=True))
    assert len(monthly) == 2
    assert np.isclose(monthly.iloc[0]["spearman"], 1.0)
    assert np.isclose(monthly.iloc[1]["spearman"], -1.0)


def test_fixed_universe_excludes_forced_extra_security():
    formation = pd.Timestamp("2012-01-31")
    crsp = pd.DataFrame(
        {
            "date": [formation] * 6,
            "permno": [1, 2, 3, 4, 5, 999],
            "market_equity": [600, 500, 400, 300, 200, 50],
            "ticker": ["A", "B", "C", "D", "E", "FORCED"],
        }
    )
    selected = fixed_universe_permnos(crsp, formation_date=formation, top_n=5)
    assert set(selected) == {1, 2, 3, 4, 5}
    assert 999 not in set(selected)


def test_timing_and_regime_validation():
    df = _prediction_panel(20)
    assignments, _ = prepare_portfolio_assignments(df)
    deciles = compute_decile_returns(assignments)
    validate_portfolio_outputs(df, assignments, deciles)
    assert assignments["regime"].eq("HIGH").all()
    assert (assignments["realized_return_date"] > assignments["formation_date"]).all()


def test_portfolio_summary_uses_sample_standard_deviation():
    first = _prediction_panel(20, date="2020-01-31")
    second = _prediction_panel(20, date="2020-02-29")
    second["actual_next_month_return"] *= 2
    assignments, _ = prepare_portfolio_assignments(pd.concat([first, second], ignore_index=True))
    deciles = compute_decile_returns(assignments)
    ls = long_short_returns(deciles)
    summary = summarize_portfolios(ls)
    row = summary[(summary["model"] == "elastic_net") & (summary["regime"] == "OVERALL") & (summary["weighting"] == "EW")].iloc[0]
    values = ls[(ls["model"] == "elastic_net") & (ls["weighting"] == "EW")]["D10_minus_D1"]
    assert np.isclose(row["sd_monthly_D10_minus_D1"], values.std(ddof=1))

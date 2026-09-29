from pathlib import Path

import numpy as np
import pandas as pd

from rdsrp.baseline.economic_value_three_regime import (
    assign_deterministic_deciles,
    attach_volatility_regimes,
    build_expanding_vix_regimes,
    classify_three_state_vix,
    build_portfolio_assignments,
    monthly_decile_returns,
    monotonicity_diagnostics,
    decile_return_profiles,
    summarize_portfolios,
)
from rdsrp.regimes.vix_regime import label_vix_regime
import rdsrp.baseline.economic_value_three_regime as extension


def _history():
    return pd.DataFrame(
        {
            "date": pd.date_range("2019-01-31", periods=18, freq="ME"),
            "vix": [10, 12, 11, 15, 18, 14, 13, 20, 16, 17, 19, 21, 22, 18, 24, 16, 25, 20],
        }
    )


def _predictions():
    history = _history()
    regimes = build_expanding_vix_regimes(history)
    oos = regimes.iloc[12:].reset_index(drop=True)
    rows = []
    for date_index, month in oos.iterrows():
        date = month["date"]
        for permno in range(1, 101):
            score = (permno - 50.5) / 50.0
            realized = 0.01 * score + 0.0001 * date_index
            rows.append(
                {
                    "date": date,
                    "formation_date": date,
                    "realized_return_date": date + pd.offsets.MonthEnd(1),
                    "permno": permno,
                    "ticker": f"S{permno}",
                    "regime": month["regime_binary"],
                    "vix": month["vix"],
                    "actual_next_month_return": realized,
                    "me_lag": float(100 + permno),
                    "me_lag_date": date + pd.offsets.MonthEnd(-1),
                    "elastic_net_prediction": score,
                    "xgboost_prediction": score**3,
                }
            )
    return pd.DataFrame(rows), history


def test_expanding_terciles_are_future_invariant_and_ordered():
    history = _history()
    original = build_expanding_vix_regimes(history)
    changed = history.copy()
    changed.loc[17, "vix"] = 999.0
    future_changed = build_expanding_vix_regimes(changed)
    pd.testing.assert_series_equal(
        original.loc[:16, "vix_expanding_q33"],
        future_changed.loc[:16, "vix_expanding_q33"],
    )
    pd.testing.assert_series_equal(
        original.loc[:16, "vix_expanding_q67"],
        future_changed.loc[:16, "vix_expanding_q67"],
    )
    pd.testing.assert_series_equal(
        original.loc[:16, "regime_3state"],
        future_changed.loc[:16, "regime_3state"],
    )
    assert (original["vix_expanding_q33"] <= original["vix_expanding_q67"]).all()


def test_three_state_boundaries_are_low_middle_high():
    assert classify_three_state_vix(10.0, 10.0, 20.0) == "LOW"
    assert classify_three_state_vix(15.0, 10.0, 20.0) == "MIDDLE"
    assert classify_three_state_vix(20.0, 10.0, 20.0) == "MIDDLE"
    assert classify_three_state_vix(20.0001, 10.0, 20.0) == "HIGH"

    history = pd.DataFrame(
        {"date": pd.date_range("2020-01-31", periods=3, freq="ME"), "vix": [10.0, 20.0, 30.0]}
    )
    out = build_expanding_vix_regimes(history)
    assert out.loc[0, "regime_3state"] == "LOW"
    assert set(out["regime_3state"]) <= {"LOW", "MIDDLE", "HIGH"}


def test_binary_regime_is_identical_to_existing_expanding_median_rule():
    history = _history()
    old = label_vix_regime(history)
    new = build_expanding_vix_regimes(history)
    assert old["regime"].tolist() == new["regime_binary"].tolist()
    assert np.allclose(old["vix_expanding_median"], new["vix_expanding_median"])


def test_deterministic_tie_breaking_uses_permno_order():
    group = pd.DataFrame({"permno": list(range(20, 0, -1)), "prediction": [1.0] * 20})
    deciles = assign_deterministic_deciles(group, "prediction")
    ordered = group.assign(decile=deciles).sort_values("permno")
    assert ordered.iloc[0]["decile"] == 1
    assert ordered.iloc[-1]["decile"] == 10
    assert ordered.groupby("decile").size().tolist() == [2] * 10


def test_full_extension_preserves_binary_labels_and_builds_ew_vw_deciles():
    predictions, history = _predictions()
    augmented, regimes = attach_volatility_regimes(predictions, history)
    assert augmented["regime"].astype(str).str.upper().tolist() == augmented["regime_binary"].astype(str).str.upper().tolist()
    assert not augmented["regime_3state"].isna().any()
    assert (regimes["vix_expanding_q33"] <= regimes["vix_expanding_q67"]).all()

    assignments = build_portfolio_assignments(augmented)
    assert set(assignments["decile"]) == set(range(1, 11))
    monthly = monthly_decile_returns(assignments)
    assert set(monthly["weighting"]) == {"EW", "VW"}
    assert np.allclose(monthly["D10_minus_D1"], monthly["D10"] - monthly["D1"])
    assert (monthly["n_stocks"] > 0).all()

    overall = summarize_portfolios(monthly, "overall")
    three = summarize_portfolios(monthly, "three_state")
    assert len(overall) == 4
    assert set(three["regime"]).issubset({"LOW", "MIDDLE", "HIGH"})

    profiles = decile_return_profiles(monthly)
    diagnostics = monotonicity_diagnostics(profiles)
    assert (diagnostics["adjacent_comparisons"] == 9).all()


def test_equal_and_value_weighted_decile_arithmetic_is_exact():
    rows = []
    date = pd.Timestamp("2024-01-31")
    for decile in range(1, 11):
        for j in range(2):
            ret = decile / 100.0 + j / 1000.0
            weight = 1.0 if j == 0 else 3.0
            rows.append(
                {
                    "formation_date": date,
                    "realized_return_date": date + pd.offsets.MonthEnd(1),
                    "permno": decile * 10 + j,
                    "model": "Elastic Net",
                    "decile": decile,
                    "realized_next_month_return": ret,
                    "weighting_variable": weight,
                    "regime_binary": "LOW",
                    "regime_3state": "MIDDLE",
                }
            )
    assignments = pd.DataFrame(rows)
    second = assignments.copy()
    second["model"] = "XGBoost"
    monthly = monthly_decile_returns(pd.concat([assignments, second], ignore_index=True))
    en = monthly[monthly["model"] == "Elastic Net"].set_index("weighting")
    assert np.isclose(en.loc["EW", "D1"], (0.010 + 0.011) / 2.0)
    assert np.isclose(en.loc["VW", "D1"], (1.0 * 0.010 + 3.0 * 0.011) / 4.0)
    assert np.isclose(en.loc["EW", "D10"], (0.100 + 0.101) / 2.0)
    assert np.isclose(en.loc["VW", "D10"], (1.0 * 0.100 + 3.0 * 0.101) / 4.0)
    assert np.isclose(en.loc["EW", "D10_minus_D1"], en.loc["EW", "D10"] - en.loc["EW", "D1"])


def test_runner_writes_required_artifact_contract_without_retraining(tmp_path, monkeypatch):
    predictions, history = _predictions()
    prediction_path = tmp_path / "results" / "baseline_models" / "baseline_oos_predictions.parquet"
    panel_path = tmp_path / "data" / "processed" / "baseline_modeling_panel.parquet"
    prediction_path.parent.mkdir(parents=True)
    panel_path.parent.mkdir(parents=True)
    prediction_path.touch()
    panel_path.touch()

    monkeypatch.setattr(extension.pd, "read_parquet", lambda path, *args, **kwargs: predictions.copy())
    monkeypatch.setattr(extension, "_read_canonical_vix_history", lambda path: history.copy())
    artifacts = extension.run_three_regime_economic_value(tmp_path)
    out = artifacts["output_dir"]
    required = {
        "decile_returns_monthly.csv",
        "long_short_returns.csv",
        "portfolio_assignments.csv",
        "portfolio_summary_overall.csv",
        "portfolio_summary_binary_regime.csv",
        "portfolio_summary_three_regime.csv",
        "three_regime_comparison.csv",
        "three_regime_economic_value_summary.csv",
        "rank_ic_by_volatility_regime.csv",
        "decile_return_profiles.csv",
        "monotonicity_diagnostics.csv",
        "regime_transition_counts.csv",
        "regime_validation.csv",
        "vix_regime_monthly.csv",
        "economic_value_summary.md",
        "three_regime_long_short_ew.png",
        "three_regime_long_short_vw.png",
        "decile_profile_overall.png",
        "cumulative_long_short_ew.png",
        "cumulative_long_short_vw.png",
        "vix_three_regimes.png",
        "run_metadata.csv",
    }
    assert required.issubset({path.name for path in out.iterdir()})
    metadata = pd.read_csv(out / "run_metadata.csv")
    assert not bool(metadata.loc[0, "models_retrained"])
    assert metadata.loc[0, "quantile_method"] == "linear"


def test_value_weight_and_realization_timing_audits_fail_hard():
    predictions, history = _predictions()
    bad_weight = predictions.copy()
    bad_weight.loc[0, "me_lag_date"] = bad_weight.loc[0, "formation_date"]
    try:
        attach_volatility_regimes(bad_weight, history)
    except AssertionError as exc:
        assert "previous calendar month-end" in str(exc)
    else:
        raise AssertionError("Expected value-weight timing audit to fail.")

    bad_realized = predictions.copy()
    bad_realized.loc[0, "realized_return_date"] = bad_realized.loc[0, "formation_date"]
    try:
        attach_volatility_regimes(bad_realized, history)
    except AssertionError as exc:
        assert "calendar month t+1" in str(exc)
    else:
        raise AssertionError("Expected realization timing audit to fail.")

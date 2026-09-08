import numpy as np
import pandas as pd
import pytest

from rdsrp.baseline import (
    BASELINE_PREDICTORS,
    baseline_model_metrics,
    deterministic_tuning_sample,
    make_validation_panel,
    monthly_rank_correlations,
    prediction_diagnostics,
    prepare_baseline_sample,
    production_count_audit,
    run_expanding_baseline_oos,
    validate_prediction_panel,
)
from rdsrp.models.elastic_net import ElasticNetModel
from rdsrp.models.tuning import forward_chaining_splits
from rdsrp.models.xgboost_model import XGBoostModel


def _models(seed=42):
    return {
        "elastic_net": ElasticNetModel(
            {"alpha": 0.001, "l1_ratio": 0.5, "max_iter": 10_000}, random_state=seed
        ),
        "xgboost": XGBoostModel(
            {
                "n_estimators": 20,
                "max_depth": 2,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "n_jobs": 1,
                "tree_method": "hist",
            },
            random_state=seed,
        ),
    }


def test_baseline_predictor_set_is_exactly_frozen_three():
    assert BASELINE_PREDICTORS == ("log_me", "book_to_market", "mom_12_2")


def test_prepare_sample_preserves_book_to_market_and_drops_missing_predictors():
    panel = make_validation_panel(n_stocks=5, n_months=20)
    original = panel.set_index(["permno", "date"])["book_to_market"].copy()
    panel.loc[0, "mom_12_2"] = np.nan
    sample = prepare_baseline_sample(panel)
    assert len(sample) == len(panel) - 1
    restored = sample.set_index(["permno", "date"])["book_to_market"]
    expected = original.loc[restored.index]
    np.testing.assert_allclose(restored.to_numpy(), expected.to_numpy())


def test_accounting_timing_column_cannot_be_later_than_formation():
    panel = make_validation_panel(n_stocks=2, n_months=20)
    panel["accounting_available_date"] = panel["date"]
    panel.loc[0, "accounting_available_date"] = panel.loc[0, "date"] + pd.offsets.MonthEnd(1)
    with pytest.raises(AssertionError, match="Accounting timing violation"):
        prepare_baseline_sample(panel)


def test_calendar_adjacency_is_required_for_target():
    panel = make_validation_panel(n_stocks=2, n_months=20)
    panel.loc[0, "realized_return_date"] = panel.loc[0, "date"] + pd.offsets.MonthEnd(2)
    with pytest.raises(AssertionError, match="calendar-adjacent"):
        prepare_baseline_sample(panel)


def test_count_audit_stops_on_mismatch_and_detects_duplicates():
    panel = make_validation_panel(n_stocks=3, n_months=20)
    audit = production_count_audit(panel, expected=None, strict=False)
    assert audit["total_stock_months"] == 60
    assert audit["complete_predictor_rows"] == 60
    assert audit["final_screened_rows"] == 60
    dup = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    observed = production_count_audit(dup, expected=None, strict=False)
    assert observed["duplicate_permno_date_rows"] == 1
    with pytest.raises(RuntimeError, match="counts differ"):
        production_count_audit(panel, expected={"total_stock_months": 999}, strict=True)


def test_forward_chaining_validation_dates_strictly_follow_training_dates():
    dates = pd.Series(pd.date_range("2010-01-31", periods=24, freq="ME").repeat(4))
    splits = list(forward_chaining_splits(dates, n_splits=4, min_train_months=12))
    assert splits
    for train_idx, valid_idx in splits:
        assert dates.iloc[train_idx].max() < dates.iloc[valid_idx].min()


def test_tuning_subsample_is_deterministic_and_historical_only():
    frame = make_validation_panel(n_stocks=20, n_months=20)
    a = deterministic_tuning_sample(frame, max_rows_per_month=5, random_state=42)
    b = deterministic_tuning_sample(frame, max_rows_per_month=5, random_state=42)
    pd.testing.assert_frame_equal(a, b)
    assert a.groupby("date").size().max() == 5
    assert a["date"].min() == frame["date"].min()
    assert a["date"].max() == frame["date"].max()


def test_elastic_net_scaler_is_fit_on_training_data_only():
    train_x = np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0], [4.0, 5.0, 6.0]])
    train_y = np.array([0.0, 0.1, 0.2])
    extreme_test = np.array([[1e9, 1e9, 1e9]])
    model = ElasticNetModel({"alpha": 0.01, "l1_ratio": 0.5}, random_state=42)
    fit = model.fit(train_x, train_y)
    _ = model.predict(fit, extreme_test)
    scaler = fit.estimator.named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, train_x.mean(axis=0))


def test_oos_timing_benchmark_regime_and_uniqueness():
    panel = make_validation_panel(n_stocks=8, n_months=24, random_state=7)
    sample = prepare_baseline_sample(panel)
    start = sorted(sample["date"].unique())[-3]
    out, _, _ = run_expanding_baseline_oos(
        sample,
        _models(seed=7),
        start_test=start,
        end_test=sample["date"].max(),
        min_train_months=12,
        retrain_every_months=1,
    )
    validate_prediction_panel(out)
    assert not out.duplicated(["permno", "date"]).any()
    assert (out["train_feature_end_date"] < out["date"]).all()
    assert (out["train_target_end_date"] <= out["date"]).all()
    assert (out["realized_return_date"] == out["date"] + pd.offsets.MonthEnd(1)).all()
    expected_regime = sample[["date", "regime"]].drop_duplicates().set_index("date")["regime"]
    assert out["regime"].tolist() == out["date"].map(expected_regime).tolist()

    first_date = pd.Timestamp(out["date"].min())
    prior = sample[(sample["date"] < first_date) & (sample["realized_return_date"] <= first_date)]
    expected_benchmark = float(prior["next_month_return"].mean())
    assert np.allclose(
        out.loc[out["date"] == first_date, "benchmark_prediction"].to_numpy(), expected_benchmark
    )


def test_future_returns_do_not_change_first_oos_benchmark():
    panel = make_validation_panel(n_stocks=6, n_months=24, random_state=9)
    sample = prepare_baseline_sample(panel)
    start = sorted(sample["date"].unique())[-2]
    base, _, _ = run_expanding_baseline_oos(
        sample,
        _models(seed=9),
        start_test=start,
        end_test=start,
        min_train_months=12,
        retrain_every_months=1,
    )
    mutated = sample.copy()
    mutated.loc[mutated["date"] > pd.Timestamp(start), "next_month_return"] = 999.0
    alt, _, _ = run_expanding_baseline_oos(
        mutated,
        _models(seed=9),
        start_test=start,
        end_test=start,
        min_train_months=12,
        retrain_every_months=1,
    )
    np.testing.assert_allclose(base["benchmark_prediction"], alt["benchmark_prediction"])


def test_fixed_seed_reproduces_predictions():
    sample = prepare_baseline_sample(make_validation_panel(n_stocks=6, n_months=22, random_state=4))
    start = sorted(sample["date"].unique())[-2]
    a, _, _ = run_expanding_baseline_oos(
        sample,
        _models(seed=123),
        start_test=start,
        end_test=start,
        min_train_months=12,
        retrain_every_months=1,
    )
    b, _, _ = run_expanding_baseline_oos(
        sample,
        _models(seed=123),
        start_test=start,
        end_test=start,
        min_train_months=12,
        retrain_every_months=1,
    )
    np.testing.assert_allclose(a["elastic_net_prediction"], b["elastic_net_prediction"])
    np.testing.assert_allclose(a["xgboost_prediction"], b["xgboost_prediction"])


def test_metrics_rank_and_diagnostics_cover_both_models_and_regimes():
    sample = prepare_baseline_sample(make_validation_panel(n_stocks=8, n_months=24, random_state=2))
    start = sorted(sample["date"].unique())[-4]
    out, _, _ = run_expanding_baseline_oos(
        sample,
        _models(seed=2),
        start_test=start,
        end_test=sample["date"].max(),
        min_train_months=12,
        retrain_every_months=1,
    )
    metrics = baseline_model_metrics(out)
    assert set(metrics["model"]) == {"Elastic Net", "XGBoost"}
    assert {"OVERALL", "HIGH", "LOW"}.issubset(set(metrics["regime"]))
    assert {"mse", "rmse", "mae", "oos_r2", "correlation", "normalized_rmse"}.issubset(
        metrics.columns
    )
    monthly = monthly_rank_correlations(out)
    assert {"spearman_ic", "pearson_ic"}.issubset(monthly.columns)
    diagnostics = prediction_diagnostics(out)
    assert len(diagnostics) == 6
    assert (diagnostics["prediction_coverage"] == 1.0).all()


def test_saved_vix_regime_must_match_expanding_median_rule():
    panel = make_validation_panel(n_stocks=3, n_months=20)
    monthly = panel[["date", "vix", "regime"]].drop_duplicates("date").sort_values("date")
    monthly["vix_expanding_median"] = monthly["vix"].expanding().median()
    panel = panel.merge(monthly[["date", "vix_expanding_median"]], on="date", how="left")
    prepare_baseline_sample(panel)
    panel.loc[panel.index[5], "regime"] = (
        "LOW" if panel.loc[panel.index[5], "regime"] == "HIGH" else "HIGH"
    )
    with pytest.raises(AssertionError, match="expanding-median rule"):
        prepare_baseline_sample(panel)


def test_frozen_production_counts_match_published_validation_artifacts():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    metadata = json.loads(
        (root / "results" / "data_validation" / "production" / "build_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    flow = pd.read_csv(root / "results" / "data_validation" / "production" / "sample_flow.csv")
    final = flow.loc[flow["stage"] == "Complete predictors + target + regime"]
    assert metadata["counts"]["final_panel_rows"] == 2_140_091
    assert metadata["counts"]["complete_predictor_rows"] == 1_798_891
    assert int(final.iloc[0]["stock_month_observations"]) == 1_340_823

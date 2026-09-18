import numpy as np
import pandas as pd
import pytest

from rdsrp.data.modeling_panel_validation import (
    coverage_by_year_report,
    extreme_observation_reports,
    finalize_baseline_modeling_panel,
    missingness_report,
    model_loader_smoke_test,
    monthly_complete_case_summary,
    monthly_predictor_coverage_report,
    predictor_correlations_report,
    predictor_summary_report,
    sample_flow_report,
    target_summary_report,
    training_rows_observable_as_of,
    validate_baseline_panel,
)
from rdsrp.model_spec import BASELINE_FEATURES


def _raw_panel(n_months=24, n_stocks=4):
    dates = pd.date_range("2020-01-31", periods=n_months, freq="ME")
    rows = []
    vix_values = np.linspace(12.0, 30.0, n_months)
    medians = pd.Series(vix_values).expanding().median().to_numpy()
    for m, date in enumerate(dates):
        regime = "HIGH" if vix_values[m] > medians[m] else "LOW"
        char_year = date.year if date.month >= 6 else date.year - 1
        accounting_date = pd.Timestamp(f"{char_year - 1}-12-31")
        eligible = pd.Timestamp(f"{char_year}-06-30")
        for permno in range(1, n_stocks + 1):
            prc = 10.0 + permno + m * 0.1
            shrout = 1_000.0 + permno
            me = prc * shrout
            be = 5.0 + permno
            me_dec = 10.0 + permno
            rows.append(
                {
                    "permno": permno,
                    "ticker": f"S{permno}",
                    "gvkey": f"G{permno}",
                    "date": date,
                    "realized_return_date": date + pd.offsets.MonthEnd(1),
                    "ret": 0.01,
                    "next_month_return": 0.01 + 0.0001 * permno,
                    "prc": prc,
                    "shrout": shrout,
                    "market_equity": me,
                    "me": me,
                    "log_me": np.log(me),
                    "me_lag": me * 0.99,
                    "accounting_datadate": accounting_date,
                    "fyear": char_year - 1,
                    "characteristic_year": char_year,
                    "accounting_available_from": eligible,
                    "book_equity": be,
                    "me_dec": me_dec,
                    "book_to_market": be / me_dec,
                    "mom_12_2": 0.05 + 0.001 * m + 0.002 * permno,
                    "vix": vix_values[m],
                    "vix_expanding_median": medians[m],
                    "regime": regime,
                    "price_eligible": True,
                }
            )
    return pd.DataFrame(rows)


def _final_panel():
    return finalize_baseline_modeling_panel(_raw_panel())


def _reference(panel):
    return panel[
        ["formation_date", "vix", "vix_expanding_median", "regime"]
    ].drop_duplicates("formation_date")


def test_contract_and_complete_case_are_explicit():
    panel = _final_panel()
    assert tuple(BASELINE_FEATURES) == ("log_me", "book_to_market", "mom_12_2")
    assert {"formation_date", "target_available_date", "baseline_complete_case"}.issubset(panel)
    assert panel["baseline_complete_case"].all()
    validate_baseline_panel(panel, reference_regimes=_reference(panel))


def test_duplicate_security_month_fails_loudly():
    panel = _final_panel()
    dup = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="duplicate security-month"):
        validate_baseline_panel(dup, reference_regimes=_reference(panel))


def test_target_must_be_exactly_next_calendar_month():
    panel = _final_panel()
    panel.loc[0, "realized_return_date"] = (
        panel.loc[0, "formation_date"] + pd.offsets.MonthEnd(2)
    )
    with pytest.raises(AssertionError, match=r"calendar t\+1"):
        validate_baseline_panel(panel, reference_regimes=_reference(panel))


def test_unobserved_target_cannot_enter_training():
    panel = _final_panel()
    cutoff = pd.Timestamp("2020-06-30")
    mask = training_rows_observable_as_of(panel, cutoff)
    assert (panel.loc[mask, "formation_date"] < cutoff).all()
    assert (panel.loc[mask, "realized_return_date"] <= cutoff).all()
    may = panel["formation_date"].eq(pd.Timestamp("2020-05-31"))
    june = panel["formation_date"].eq(cutoff)
    assert mask[may].all()
    assert not mask[june].any()


def test_nonpositive_me_cannot_have_valid_log_me():
    panel = _final_panel()
    panel.loc[0, "shrout"] = 0.0
    panel.loc[0, "me"] = 0.0
    panel.loc[0, "market_equity"] = 0.0
    panel.loc[0, "log_me"] = 0.0
    with pytest.raises(AssertionError, match="Nonpositive market equity"):
        validate_baseline_panel(panel, reference_regimes=_reference(panel))


def test_future_accounting_fails_and_june_characteristic_year_is_enforced():
    panel = _final_panel()
    panel.loc[0, "accounting_eligibility_date"] = (
        panel.loc[0, "formation_date"] + pd.offsets.MonthEnd(1)
    )
    with pytest.raises(AssertionError, match="Future accounting information"):
        validate_baseline_panel(panel, reference_regimes=_reference(panel))

    panel = _final_panel()
    june = panel.index[panel["formation_date"].eq(pd.Timestamp("2020-06-30"))][0]
    panel.loc[june, "characteristic_year"] = 2019
    with pytest.raises(AssertionError, match="June/May"):
        validate_baseline_panel(panel, reference_regimes=_reference(panel))


def test_book_to_market_units_identity_is_enforced():
    panel = _final_panel()
    panel.loc[0, "book_to_market"] *= 1000.0
    with pytest.raises(AssertionError, match="book_equity / me_dec"):
        validate_baseline_panel(panel, reference_regimes=_reference(panel))


def test_price_filter_uses_formation_price():
    raw = _raw_panel()
    raw.loc[0, "prc"] = 4.0
    raw.loc[0, "market_equity"] = 4.0 * raw.loc[0, "shrout"]
    raw.loc[0, "me"] = raw.loc[0, "market_equity"]
    raw.loc[0, "log_me"] = np.log(raw.loc[0, "me"])
    raw.loc[0, "price_eligible"] = False
    panel = finalize_baseline_modeling_panel(raw)
    target = panel[
        panel["permno"].eq(1)
        & panel["formation_date"].eq(pd.Timestamp("2020-01-31"))
    ]
    assert not target["baseline_complete_case"].iloc[0]
    validate_baseline_panel(panel, reference_regimes=_reference(panel))

    panel.loc[0, "price_eligible"] = True
    with pytest.raises(AssertionError, match="formation-month price"):
        validate_baseline_panel(panel, reference_regimes=_reference(panel))


def test_regime_reference_mismatch_fails():
    panel = _final_panel()
    ref = _reference(panel)
    ref.loc[ref.index[-1], "vix_expanding_median"] += 1.0
    with pytest.raises(AssertionError, match="historical-only VIX reference"):
        validate_baseline_panel(panel, reference_regimes=ref)


def test_complete_case_requires_target_regime_predictors_and_price():
    raw = _raw_panel()
    raw.loc[0, "mom_12_2"] = np.nan
    raw.loc[1, "next_month_return"] = np.nan
    raw.loc[2, "regime"] = pd.NA
    raw.loc[3, "price_eligible"] = False
    panel = finalize_baseline_modeling_panel(raw)
    assert not panel.loc[:3, "baseline_complete_case"].any()


def test_reports_and_model_loader_smoke_test():
    panel = _final_panel()
    metadata = validate_baseline_panel(panel, reference_regimes=_reference(panel))
    flow = sample_flow_report(panel)
    missing = missingness_report(panel)
    summary = predictor_summary_report(panel)
    target = target_summary_report(panel)
    yearly = coverage_by_year_report(panel)
    monthly = monthly_predictor_coverage_report(panel)
    corr = predictor_correlations_report(panel)
    extremes, audit = extreme_observation_reports(panel, n_each_tail=2)
    monthly_summary = monthly_complete_case_summary(monthly)
    smoke = model_loader_smoke_test(panel)
    assert flow.iloc[-1]["Observations"] == len(panel)
    assert not missing.empty
    assert set(summary["variable"]) == set(BASELINE_FEATURES)
    assert target.iloc[0]["variable"] == "next_month_return"
    assert not yearly.empty and not monthly.empty and not corr.empty
    assert not extremes.empty and not audit.empty
    assert monthly_summary["minimum"] == 4
    assert smoke["elastic_net"] == "pass" and smoke["xgboost"] == "pass"
    assert metadata["duplicate_security_month_rows"] == 0


def test_quick_and_production_share_baseline_contract():
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    quick = yaml.safe_load((root / "configs" / "quick.yaml").read_text(encoding="utf-8"))
    production = yaml.safe_load(
        (root / "configs" / "production.yaml").read_text(encoding="utf-8")
    )
    expected = {
        "predictors": ["log_me", "book_to_market", "mom_12_2"],
        "target": "next_month_return",
        "formation_date": "formation_date",
        "realized_return_date": "realized_return_date",
    }
    assert quick["run"]["baseline"] == expected
    assert production["run"]["baseline"] == expected

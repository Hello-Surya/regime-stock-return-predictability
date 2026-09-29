from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from rdsrp.inference.hac import (
    REGIME_3STATE_ORDER,
    fit_regime_regression,
    hac_mean,
    regime_difference_from_fit,
    significance_stars,
)
from rdsrp.inference.multiple_testing import add_multiple_testing_adjustments
from rdsrp.inference.pipeline import (
    load_authoritative_inputs,
    long_short_hac_inference,
    long_short_regime_differences,
    rank_ic_hac_inference,
    rank_ic_regime_differences,
)


def _monthly_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2010-01-31", periods=36, freq="ME")
    regimes = np.resize(np.array(["LOW", "MIDDLE", "HIGH"]), len(dates))
    binary = np.where(regimes == "HIGH", "HIGH", "LOW")
    ls_rows = []
    ic_rows = []
    offsets = {"LOW": 0.002, "MIDDLE": 0.005, "HIGH": 0.009}
    for model_i, model in enumerate(("Elastic Net", "XGBoost")):
        ic = np.array([offsets[r] + 0.0002 * np.sin(i) + model_i * 0.001 for i, r in enumerate(regimes)])
        for i, date in enumerate(dates):
            ic_rows.append({"formation_date": date, "model": model, "regime_binary": binary[i], "regime_3state": regimes[i], "spearman_ic": ic[i]})
        for weighting_i, weighting in enumerate(("EW", "VW")):
            values = np.array([offsets[r] + 0.0003 * np.cos(i) + 0.001 * model_i - 0.0005 * weighting_i for i, r in enumerate(regimes)])
            for i, date in enumerate(dates):
                ls_rows.append({"formation_date": date, "model": model, "weighting": weighting, "regime_binary": binary[i], "regime_3state": regimes[i], "D10_minus_D1": values[i]})
    return pd.DataFrame(ls_rows), pd.DataFrame(ic_rows)


def test_hac_mean_intercept_equals_sample_mean_and_ci_contains_estimate():
    values = pd.Series([0.01, 0.02, -0.01, 0.03, 0.00, 0.015, 0.005, 0.02])
    result = hac_mean(values, maxlags=2)
    assert np.isclose(result.estimate, values.mean())
    assert np.isfinite([result.hac_se, result.t_stat, result.p_value, result.ci_lower, result.ci_upper]).all()
    assert result.ci_lower <= result.estimate <= result.ci_upper


def test_group_dummy_coefficients_and_high_middle_linear_contrast_equal_direct_means():
    frame = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "regime": ["LOW", "MIDDLE", "HIGH"] * 2})
    fit, used = fit_regime_regression(frame, outcome_col="y", regime_col="regime", regime_order=REGIME_3STATE_ORDER, maxlags=1)
    middle_low = regime_difference_from_fit(fit, "MIDDLE", "LOW", regime_order=REGIME_3STATE_ORDER, n_months=len(used), hac_lag=1)
    high_middle = regime_difference_from_fit(fit, "HIGH", "MIDDLE", regime_order=REGIME_3STATE_ORDER, n_months=len(used), hac_lag=1)
    means = frame.groupby("regime")["y"].mean()
    assert np.isclose(middle_low.estimate, means["MIDDLE"] - means["LOW"])
    assert np.isclose(high_middle.estimate, means["HIGH"] - means["MIDDLE"])


def test_holm_and_bh_match_statsmodels_and_holm_is_not_below_raw():
    frame = pd.DataFrame({"model": ["X"] * 3, "raw_p_value": [0.01, 0.04, 0.20]})
    out = add_multiple_testing_adjustments(frame, family_cols=["model"])
    assert np.allclose(out["holm_p_value"], multipletests(frame["raw_p_value"], method="holm")[1])
    assert np.allclose(out["bh_fdr_p_value"], multipletests(frame["raw_p_value"], method="fdr_bh")[1])
    assert (out["holm_p_value"] >= out["raw_p_value"]).all()


def test_significance_star_thresholds_and_boundaries():
    assert significance_stars(0.005) == "***"
    assert significance_stars(0.03) == "**"
    assert significance_stars(0.08) == "*"
    assert significance_stars(0.20) == ""
    assert significance_stars(0.01) == "**"
    assert significance_stars(0.05) == "*"
    assert significance_stars(0.10) == ""


def test_three_state_outputs_preserve_models_weightings_regimes_and_month_counts():
    ls, ic = _monthly_fixture()
    means = long_short_hac_inference(ls)
    diffs = long_short_regime_differences(ls)
    rank = rank_ic_hac_inference(ic)
    rank_diffs = rank_ic_regime_differences(ic)
    assert set(means["model"]) == {"Elastic Net", "XGBoost"}
    assert set(means["weighting"]) == {"EW", "VW"}
    assert set(means.loc[means["regime_definition"] == "three_state", "regime"]) == set(REGIME_3STATE_ORDER)
    assert set(diffs["comparison"]) == {"MIDDLE - LOW", "HIGH - LOW", "HIGH - MIDDLE"}
    assert set(rank.loc[rank["regime_definition"] == "three_state", "regime"]) == set(REGIME_3STATE_ORDER)
    assert set(rank_diffs["comparison"]) == {"MIDDLE - LOW", "HIGH - LOW", "HIGH - MIDDLE"}
    assert (diffs["n_months"] == 36).all()
    assert (rank_diffs["n_months"] == 36).all()


def test_inference_means_equal_direct_descriptive_means():
    ls, ic = _monthly_fixture()
    means = long_short_hac_inference(ls)
    rank = rank_ic_hac_inference(ic)
    row = means[(means["model"] == "XGBoost") & (means["weighting"] == "EW") & (means["regime"] == "HIGH")].iloc[0]
    direct = ls[(ls["model"] == "XGBoost") & (ls["weighting"] == "EW") & (ls["regime_3state"] == "HIGH")]["D10_minus_D1"].mean()
    assert np.isclose(row["mean_monthly_return"], direct)
    rrow = rank[(rank["model"] == "Elastic Net") & (rank["regime"] == "MIDDLE")].iloc[0]
    rdirect = ic[(ic["model"] == "Elastic Net") & (ic["regime_3state"] == "MIDDLE")]["spearman_ic"].mean()
    assert np.isclose(rrow["mean_spearman_ic"], rdirect)



def test_loader_rejects_duplicate_months_and_invalid_regime_labels(tmp_path):
    ls, ic = _monthly_fixture()
    source = tmp_path / "results" / "economic_value"
    source.mkdir(parents=True)
    pd.concat([ls, ls.iloc[[0]]], ignore_index=True).to_csv(
        source / "long_short_returns.csv", index=False
    )
    ic.to_csv(source / "rank_ic_monthly.csv", index=False)
    try:
        load_authoritative_inputs(tmp_path)
    except AssertionError as exc:
        assert "Duplicate monthly inference observation" in str(exc)
    else:
        raise AssertionError("Duplicate monthly observation should fail hard.")

    ls.to_csv(source / "long_short_returns.csv", index=False)
    bad = ic.copy()
    bad.loc[0, "regime_3state"] = "OTHER"
    bad.to_csv(source / "rank_ic_monthly.csv", index=False)
    try:
        load_authoritative_inputs(tmp_path)
    except AssertionError as exc:
        assert "Unexpected stored regime labels" in str(exc)
    else:
        raise AssertionError("Unexpected regime label should fail hard.")

def test_runner_consumes_existing_monthly_artifacts_and_writes_new_outputs(tmp_path):
    from rdsrp.inference.pipeline import run_formal_inference

    ls, ic = _monthly_fixture()
    source = tmp_path / "results" / "economic_value"
    source.mkdir(parents=True)
    ls.to_csv(source / "long_short_returns.csv", index=False)
    ic.to_csv(source / "rank_ic_monthly.csv", index=False)

    overall_rows = []
    three_rows = []
    for (model, weighting), group in ls.groupby(["model", "weighting"]):
        overall_rows.append({"model": model, "weighting": weighting, "regime": "OVERALL", "mean_D10_minus_D1": group["D10_minus_D1"].mean()})
        for regime, rg in group.groupby("regime_3state"):
            three_rows.append({"model": model, "weighting": weighting, "regime": regime, "mean_D10_minus_D1": rg["D10_minus_D1"].mean()})
    pd.DataFrame(overall_rows).to_csv(source / "portfolio_summary_overall.csv", index=False)
    pd.DataFrame(three_rows).to_csv(source / "portfolio_summary_three_regime.csv", index=False)

    rank_rows = []
    for model, group in ic.groupby("model"):
        rank_rows.append({"model": model, "regime_definition": "overall", "regime": "OVERALL", "mean_spearman": group["spearman_ic"].mean()})
        for regime, rg in group.groupby("regime_3state"):
            rank_rows.append({"model": model, "regime_definition": "three_state", "regime": regime, "mean_spearman": rg["spearman_ic"].mean()})
    pd.DataFrame(rank_rows).to_csv(source / "rank_ic_by_volatility_regime.csv", index=False)

    artifacts = run_formal_inference(tmp_path)
    out = artifacts["output_dir"]
    required = {
        "long_short_hac_inference.csv",
        "long_short_regime_differences.csv",
        "rank_ic_hac_inference.csv",
        "rank_ic_regime_differences.csv",
        "binary_regime_hac_inference.csv",
        "binary_regime_difference.csv",
        "publication_inference_table.csv",
        "long_short_regime_confidence_intervals.png",
        "rank_ic_regime_confidence_intervals.png",
        "inference_summary.md",
        "run_metadata.csv",
    }
    assert required.issubset({p.name for p in out.iterdir()})
    assert (tmp_path / "paper" / "generated" / "table_long_short_inference.tex").exists()
    assert (tmp_path / "paper" / "generated" / "table_rank_ic_inference.tex").exists()
    assert (tmp_path / "paper" / "generated" / "table_regime_differences.tex").exists()

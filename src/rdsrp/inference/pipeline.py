"""Formal HAC inference layered on the frozen economic-value research artifacts."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rdsrp.inference.hac import (
    BASELINE_HAC_LAG,
    BINARY_REGIME_ORDER,
    CONFIDENCE_LEVEL,
    PAIRWISE_COMPARISONS,
    REGIME_3STATE_ORDER,
    fit_regime_regression,
    hac_mean,
    regime_difference_from_fit,
    significance_stars,
)
from rdsrp.inference.multiple_testing import add_multiple_testing_adjustments
from rdsrp.inference.tables import (
    write_long_short_latex,
    write_rank_ic_latex,
    write_regime_differences_latex,
)

MODELS = ("Elastic Net", "XGBoost")
WEIGHTINGS = ("EW", "VW")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_monthly_table(
    frame: pd.DataFrame,
    *,
    date_col: str,
    key_cols: list[str],
    outcome_col: str,
    regime_cols: list[str],
) -> pd.DataFrame:
    required = {date_col, outcome_col, *key_cols, *regime_cols}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Monthly inference input is missing columns: {missing}")
    out = frame.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out[outcome_col] = pd.to_numeric(out[outcome_col], errors="coerce")
    if out[date_col].isna().any():
        raise ValueError("Monthly inference input contains invalid formation dates.")
    if not np.isfinite(out[outcome_col].dropna().to_numpy(float)).all():
        raise ValueError(f"{outcome_col} contains non-finite non-missing values.")
    for col in regime_cols:
        out[col] = out[col].astype(str).str.upper()
    duplicates = out.duplicated([date_col, *key_cols], keep=False)
    if duplicates.any():
        example = out.loc[duplicates, [date_col, *key_cols]].iloc[0].to_dict()
        raise AssertionError(f"Duplicate monthly inference observation: {example}")
    return out.sort_values([*key_cols, date_col], kind="mergesort").reset_index(drop=True)


def load_authoritative_inputs(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = repo_root / "results" / "economic_value"
    long_short_path = source / "long_short_returns.csv"
    rank_path = source / "rank_ic_monthly.csv"
    if not long_short_path.exists():
        raise FileNotFoundError(f"Monthly D10-D1 artifact is missing: {long_short_path}")
    if not rank_path.exists():
        raise FileNotFoundError(f"Monthly rank-IC artifact is missing: {rank_path}")
    long_short = pd.read_csv(long_short_path)
    rank_ic = pd.read_csv(rank_path)
    long_short = _validate_monthly_table(
        long_short,
        date_col="formation_date",
        key_cols=["model", "weighting"],
        outcome_col="D10_minus_D1",
        regime_cols=["regime_binary", "regime_3state"],
    )
    rank_ic = _validate_monthly_table(
        rank_ic,
        date_col="formation_date",
        key_cols=["model"],
        outcome_col="spearman_ic",
        regime_cols=["regime_binary", "regime_3state"],
    )
    invalid_three = set(long_short["regime_3state"].dropna()) - set(REGIME_3STATE_ORDER)
    invalid_three |= set(rank_ic["regime_3state"].dropna()) - set(REGIME_3STATE_ORDER)
    invalid_binary = set(long_short["regime_binary"].dropna()) - set(BINARY_REGIME_ORDER)
    invalid_binary |= set(rank_ic["regime_binary"].dropna()) - set(BINARY_REGIME_ORDER)
    if invalid_three or invalid_binary:
        raise AssertionError(
            f"Unexpected stored regime labels: three_state={sorted(invalid_three)}, binary={sorted(invalid_binary)}"
        )
    if set(long_short["model"]) != set(MODELS) or set(rank_ic["model"]) != set(MODELS):
        raise AssertionError("Monthly inference inputs do not contain the frozen two-model set.")
    if set(long_short["weighting"]) != set(WEIGHTINGS):
        raise AssertionError("Monthly D10-D1 input does not contain both EW and VW series.")
    for model in MODELS:
        if set(rank_ic.loc[rank_ic["model"] == model, "regime_3state"]) != set(REGIME_3STATE_ORDER):
            raise AssertionError(f"Rank-IC input is missing a three-state regime for {model}.")
        for weighting in WEIGHTINGS:
            subset = long_short[(long_short["model"] == model) & (long_short["weighting"] == weighting)]
            if set(subset["regime_3state"]) != set(REGIME_3STATE_ORDER):
                raise AssertionError(f"D10-D1 input is missing a three-state regime for {model}/{weighting}.")
    return long_short, rank_ic


def _mean_rows(
    frame: pd.DataFrame,
    *,
    outcome_col: str,
    regime_col: str,
    regime_definition: str,
    include_weighting: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["model", "weighting"] if include_weighting else ["model"]
    for key, group in frame.groupby(keys, sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(keys, key_tuple, strict=True))
        overall = hac_mean(group[outcome_col], maxlags=BASELINE_HAC_LAG)
        rows.append(
            {
                **identity,
                "regime_definition": "overall",
                "regime": "OVERALL",
                "estimate": overall.estimate,
                "hac_se": overall.hac_se,
                "t_stat": overall.t_stat,
                "p_value": overall.p_value,
                "ci_lower": overall.ci_lower,
                "ci_upper": overall.ci_upper,
                "n_months": overall.n_months,
                "hac_lag": overall.hac_lag,
                "significance": significance_stars(overall.p_value),
            }
        )
        for regime in REGIME_3STATE_ORDER:
            subset = group[group[regime_col] == regime]
            inf = hac_mean(subset[outcome_col], maxlags=BASELINE_HAC_LAG)
            rows.append(
                {
                    **identity,
                    "regime_definition": regime_definition,
                    "regime": regime,
                    "estimate": inf.estimate,
                    "hac_se": inf.hac_se,
                    "t_stat": inf.t_stat,
                    "p_value": inf.p_value,
                    "ci_lower": inf.ci_lower,
                    "ci_upper": inf.ci_upper,
                    "n_months": inf.n_months,
                    "hac_lag": inf.hac_lag,
                    "significance": significance_stars(inf.p_value),
                }
            )
    return pd.DataFrame(rows)


def long_short_hac_inference(long_short: pd.DataFrame) -> pd.DataFrame:
    out = _mean_rows(
        long_short,
        outcome_col="D10_minus_D1",
        regime_col="regime_3state",
        regime_definition="three_state",
        include_weighting=True,
    ).rename(columns={"estimate": "mean_monthly_return"})
    return out[
        [
            "model", "weighting", "regime_definition", "regime", "mean_monthly_return",
            "hac_se", "t_stat", "p_value", "ci_lower", "ci_upper", "n_months", "hac_lag",
            "significance",
        ]
    ]


def rank_ic_hac_inference(rank_ic: pd.DataFrame) -> pd.DataFrame:
    out = _mean_rows(
        rank_ic,
        outcome_col="spearman_ic",
        regime_col="regime_3state",
        regime_definition="three_state",
        include_weighting=False,
    ).rename(columns={"estimate": "mean_spearman_ic"})
    return out[
        [
            "model", "regime_definition", "regime", "mean_spearman_ic", "hac_se", "t_stat",
            "p_value", "ci_lower", "ci_upper", "n_months", "hac_lag", "significance",
        ]
    ]


def _difference_rows(
    frame: pd.DataFrame,
    *,
    outcome_col: str,
    regime_col: str,
    include_weighting: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["model", "weighting"] if include_weighting else ["model"]
    for key, group in frame.groupby(keys, sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(keys, key_tuple, strict=True))
        group = group.sort_values("formation_date", kind="mergesort")
        fit, used = fit_regime_regression(
            group,
            outcome_col=outcome_col,
            regime_col=regime_col,
            regime_order=REGIME_3STATE_ORDER,
            maxlags=BASELINE_HAC_LAG,
        )
        for left, right in PAIRWISE_COMPARISONS:
            inf = regime_difference_from_fit(
                fit,
                left,
                right,
                regime_order=REGIME_3STATE_ORDER,
                n_months=len(used),
                hac_lag=BASELINE_HAC_LAG,
            )
            rows.append(
                {
                    **identity,
                    "comparison": f"{left} - {right}",
                    "estimated_difference": inf.estimate,
                    "hac_se": inf.hac_se,
                    "t_stat": inf.t_stat,
                    "raw_p_value": inf.p_value,
                    "ci_lower": inf.ci_lower,
                    "ci_upper": inf.ci_upper,
                    "n_months": inf.n_months,
                    "hac_lag": inf.hac_lag,
                    "raw_significance": significance_stars(inf.p_value),
                }
            )
    out = pd.DataFrame(rows)
    family_cols = ["model", "weighting"] if include_weighting else ["model"]
    out = add_multiple_testing_adjustments(out, family_cols=family_cols)
    out["adjusted_significance"] = out["holm_p_value"].map(significance_stars)
    return out


def long_short_regime_differences(long_short: pd.DataFrame) -> pd.DataFrame:
    return _difference_rows(
        long_short,
        outcome_col="D10_minus_D1",
        regime_col="regime_3state",
        include_weighting=True,
    )[
        [
            "model", "weighting", "comparison", "estimated_difference", "hac_se", "t_stat",
            "raw_p_value", "holm_p_value", "bh_fdr_p_value", "ci_lower", "ci_upper", "n_months",
            "hac_lag", "raw_significance", "adjusted_significance",
        ]
    ]


def rank_ic_regime_differences(rank_ic: pd.DataFrame) -> pd.DataFrame:
    return _difference_rows(
        rank_ic,
        outcome_col="spearman_ic",
        regime_col="regime_3state",
        include_weighting=False,
    )[
        [
            "model", "comparison", "estimated_difference", "hac_se", "t_stat", "raw_p_value",
            "holm_p_value", "bh_fdr_p_value", "ci_lower", "ci_upper", "n_months", "hac_lag",
            "raw_significance", "adjusted_significance",
        ]
    ]


def _binary_inference(frame: pd.DataFrame, *, outcome_col: str, include_weighting: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["model", "weighting"] if include_weighting else ["model"]
    mean_rows: list[dict[str, Any]] = []
    diff_rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(keys, sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(keys, key_tuple, strict=True))
        group = group.sort_values("formation_date", kind="mergesort")
        for regime in BINARY_REGIME_ORDER:
            inf = hac_mean(group.loc[group["regime_binary"] == regime, outcome_col], maxlags=BASELINE_HAC_LAG)
            mean_rows.append({**identity, "regime": regime, "estimate": inf.estimate, "hac_se": inf.hac_se,
                              "t_stat": inf.t_stat, "p_value": inf.p_value, "ci_lower": inf.ci_lower,
                              "ci_upper": inf.ci_upper, "n_months": inf.n_months, "hac_lag": inf.hac_lag,
                              "significance": significance_stars(inf.p_value)})
        fit, used = fit_regime_regression(group, outcome_col=outcome_col, regime_col="regime_binary",
                                          regime_order=BINARY_REGIME_ORDER, maxlags=BASELINE_HAC_LAG)
        diff = regime_difference_from_fit(fit, "HIGH", "LOW", regime_order=BINARY_REGIME_ORDER,
                                          n_months=len(used), hac_lag=BASELINE_HAC_LAG)
        diff_rows.append({**identity, "comparison": "HIGH - LOW", "estimated_difference": diff.estimate,
                          "hac_se": diff.hac_se, "t_stat": diff.t_stat, "p_value": diff.p_value,
                          "ci_lower": diff.ci_lower, "ci_upper": diff.ci_upper, "n_months": diff.n_months,
                          "hac_lag": diff.hac_lag, "significance": significance_stars(diff.p_value)})
    return pd.DataFrame(mean_rows), pd.DataFrame(diff_rows)


def validate_descriptive_consistency(repo_root: Path, long_short: pd.DataFrame, rank_ic: pd.DataFrame) -> None:
    base = repo_root / "results" / "economic_value"
    overall = pd.read_csv(base / "portfolio_summary_overall.csv")
    three = pd.read_csv(base / "portfolio_summary_three_regime.csv")
    rank_summary = pd.read_csv(base / "rank_ic_by_volatility_regime.csv")
    tol = {"rtol": 1e-10, "atol": 1e-12}
    for row in overall.itertuples(index=False):
        sample = long_short[(long_short["model"] == row.model) & (long_short["weighting"] == row.weighting)]
        observed = float(sample["D10_minus_D1"].mean())
        if not np.isclose(observed, float(row.mean_D10_minus_D1), **tol):
            raise AssertionError(f"Overall D10-D1 mean mismatch for {row.model}/{row.weighting}.")
        if hasattr(row, "n_months") and int(sample["D10_minus_D1"].notna().sum()) != int(row.n_months):
            raise AssertionError(f"Overall D10-D1 month-count mismatch for {row.model}/{row.weighting}.")
    for row in three.itertuples(index=False):
        sample = long_short[(long_short["model"] == row.model) & (long_short["weighting"] == row.weighting) &
                            (long_short["regime_3state"] == row.regime)]
        observed = float(sample["D10_minus_D1"].mean())
        if not np.isclose(observed, float(row.mean_D10_minus_D1), **tol):
            raise AssertionError(f"Three-state D10-D1 mean mismatch for {row.model}/{row.weighting}/{row.regime}.")
        if hasattr(row, "n_months") and int(sample["D10_minus_D1"].notna().sum()) != int(row.n_months):
            raise AssertionError(f"Three-state D10-D1 month-count mismatch for {row.model}/{row.weighting}/{row.regime}.")
    desired = rank_summary[rank_summary["regime_definition"].isin(["overall", "three_state"])]
    for row in desired.itertuples(index=False):
        sample = rank_ic[rank_ic["model"] == row.model]
        if row.regime_definition == "three_state":
            sample = sample[sample["regime_3state"] == row.regime]
        observed = float(sample["spearman_ic"].mean())
        if not np.isclose(observed, float(row.mean_spearman), **tol):
            raise AssertionError(f"Rank-IC mean mismatch for {row.model}/{row.regime_definition}/{row.regime}.")
        if hasattr(row, "n_months") and int(sample["spearman_ic"].notna().sum()) != int(row.n_months):
            raise AssertionError(f"Rank-IC month-count mismatch for {row.model}/{row.regime_definition}/{row.regime}.")


def _confidence_interval_plot(frame: pd.DataFrame, *, estimate_col: str, output_path: Path, title: str) -> None:
    view = frame[frame["regime_definition"] == "three_state"].copy()
    labels = []
    for row in view.itertuples(index=False):
        weight = f" {row.weighting}" if hasattr(row, "weighting") else ""
        labels.append(f"{row.model}{weight} {row.regime}")
    y = np.arange(len(view))
    estimates = view[estimate_col].to_numpy(float)
    lower = estimates - view["ci_lower"].to_numpy(float)
    upper = view["ci_upper"].to_numpy(float) - estimates
    fig, ax = plt.subplots(figsize=(9, max(5.0, len(view) * 0.38)))
    ax.errorbar(estimates, y, xerr=np.vstack([lower, upper]), fmt="o", capsize=3)
    ax.axvline(0.0, linewidth=1.0)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Estimate with 95% HAC confidence interval")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _summary_markdown(ls: pd.DataFrame, lsd: pd.DataFrame, ic: pd.DataFrame, icd: pd.DataFrame) -> str:
    parts = [
        "# Formal Statistical Inference",
        "",
        "## HAC methodology",
        f"Monthly inference uses Newey--West/HAC covariance with the repository's frozen {BASELINE_HAC_LAG}-month lag and {int(CONFIDENCE_LEVEL * 100)}% confidence intervals.",
        "Three-state regime differences use one LOW-reference dummy regression per model/weighting (or model for IC), with covariance-consistent linear contrasts.",
        "Holm adjustment is primary across the three pre-specified pairwise contrasts within each model/outcome/weighting family; Benjamini--Hochberg FDR p-values are retained as a secondary diagnostic.",
        "",
        "## D10-D1 inference",
    ]
    for row in ls[ls["regime_definition"] == "three_state"].itertuples(index=False):
        parts.append(f"- {row.model} {row.weighting} {row.regime}: mean={row.mean_monthly_return:.5f}, HAC SE={row.hac_se:.5f}, t={row.t_stat:.2f}, p={row.p_value:.4f}, 95% CI=[{row.ci_lower:.5f}, {row.ci_upper:.5f}], N={int(row.n_months)}.")
    parts.extend(["", "## Regime differences"])
    for row in lsd.itertuples(index=False):
        parts.append(f"- {row.model} {row.weighting} {row.comparison}: difference={row.estimated_difference:.5f}, HAC SE={row.hac_se:.5f}, t={row.t_stat:.2f}, raw p={row.raw_p_value:.4f}, Holm p={row.holm_p_value:.4f}, N={int(row.n_months)}.")
    parts.extend(["", "## Monthly Rank IC"])
    for row in ic[ic["regime_definition"] == "three_state"].itertuples(index=False):
        parts.append(f"- {row.model} {row.regime}: mean IC={row.mean_spearman_ic:.5f}, HAC SE={row.hac_se:.5f}, t={row.t_stat:.2f}, p={row.p_value:.4f}, 95% CI=[{row.ci_lower:.5f}, {row.ci_upper:.5f}], N={int(row.n_months)}.")
    parts.extend(["", "## Rank IC regime differences"])
    for row in icd.itertuples(index=False):
        parts.append(f"- {row.model} {row.comparison}: difference={row.estimated_difference:.5f}, HAC SE={row.hac_se:.5f}, t={row.t_stat:.2f}, raw p={row.raw_p_value:.4f}, Holm p={row.holm_p_value:.4f}, N={int(row.n_months)}.")
    survivors = lsd[lsd["holm_p_value"] < 0.05]
    ic_survivors = icd[icd["holm_p_value"] < 0.05]
    parts.extend(["", "## Multiple testing"])
    if survivors.empty and ic_survivors.empty:
        parts.append("No pairwise three-state regime difference is statistically distinguishable from zero at the 5% family-wise level after Holm adjustment.")
    else:
        for row in survivors.itertuples(index=False):
            parts.append(f"- Portfolio contrast surviving Holm at 5%: {row.model} {row.weighting} {row.comparison} (Holm p={row.holm_p_value:.4f}).")
        for row in ic_survivors.itertuples(index=False):
            parts.append(f"- Rank-IC contrast surviving Holm at 5%: {row.model} {row.comparison} (Holm p={row.holm_p_value:.4f}).")
    parts.extend([
        "", "## Interpretation",
        "Inference is conditional on the frozen empirical specification. Statistical significance is not interpreted as economic importance, and regime associations are observational rather than causal.",
        "", "## Limitations",
        "Multiple-testing control is defined only for the pre-specified three-contrast families reported here. Lag sensitivity, alternative regime thresholds, turnover, and richer transaction-cost analysis belong to later robustness stages.",
        "",
    ])
    return "\n".join(parts)


def run_formal_inference(repo_root: Path) -> dict[str, Path]:
    output_dir = repo_root / "results" / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = repo_root / "results" / "economic_value"
    protected_inputs = [
        source_dir / "long_short_returns.csv",
        source_dir / "rank_ic_monthly.csv",
        source_dir / "portfolio_summary_overall.csv",
        source_dir / "portfolio_summary_three_regime.csv",
        source_dir / "rank_ic_by_volatility_regime.csv",
    ]
    before_hashes = {path: _sha256(path) for path in protected_inputs}
    print("[1/8] Loading validated monthly portfolio and rank-IC series...")
    long_short, rank_ic = load_authoritative_inputs(repo_root)
    print("[2/8] Validating descriptive-output consistency...")
    validate_descriptive_consistency(repo_root, long_short, rank_ic)
    print("[3/8] Estimating HAC inference for D10-D1...")
    ls = long_short_hac_inference(long_short)
    print("[4/8] Testing LOW/MIDDLE/HIGH regime differences...")
    lsd = long_short_regime_differences(long_short)
    print("[5/8] Estimating HAC inference for monthly rank IC...")
    ic = rank_ic_hac_inference(rank_ic)
    icd = rank_ic_regime_differences(rank_ic)
    print("[6/8] Applying multiple-testing adjustments...")
    # Adjustments are applied inside the difference builders to freeze family definitions.
    print("[7/8] Generating publication tables and confidence intervals...")
    ls.to_csv(output_dir / "long_short_hac_inference.csv", index=False)
    lsd.to_csv(output_dir / "long_short_regime_differences.csv", index=False)
    ic.to_csv(output_dir / "rank_ic_hac_inference.csv", index=False)
    icd.to_csv(output_dir / "rank_ic_regime_differences.csv", index=False)
    binary_ls, binary_lsd = _binary_inference(long_short, outcome_col="D10_minus_D1", include_weighting=True)
    binary_ic, binary_icd = _binary_inference(rank_ic, outcome_col="spearman_ic", include_weighting=False)
    binary_ls.insert(0, "outcome", "D10_minus_D1")
    binary_ic.insert(0, "outcome", "spearman_ic")
    pd.concat([binary_ls, binary_ic], ignore_index=True, sort=False).to_csv(output_dir / "binary_regime_hac_inference.csv", index=False)
    binary_lsd.insert(0, "outcome", "D10_minus_D1")
    binary_icd.insert(0, "outcome", "spearman_ic")
    pd.concat([binary_lsd, binary_icd], ignore_index=True, sort=False).to_csv(output_dir / "binary_regime_difference.csv", index=False)
    publication = pd.concat([
        ls.assign(panel="D10-D1 mean inference"),
        lsd.assign(panel="D10-D1 regime differences"),
        ic.assign(panel="Rank IC mean inference"),
        icd.assign(panel="Rank IC regime differences"),
    ], ignore_index=True, sort=False)
    publication.to_csv(output_dir / "publication_inference_table.csv", index=False)
    paper_generated = repo_root / "paper" / "generated"
    write_long_short_latex(ls, lsd, paper_generated / "table_long_short_inference.tex")
    write_rank_ic_latex(ic, icd, paper_generated / "table_rank_ic_inference.tex")
    write_regime_differences_latex(
        lsd, icd, paper_generated / "table_regime_differences.tex"
    )
    _confidence_interval_plot(ls, estimate_col="mean_monthly_return", output_path=output_dir / "long_short_regime_confidence_intervals.png", title="D10-D1 means by three-state VIX regime")
    _confidence_interval_plot(ic, estimate_col="mean_spearman_ic", output_path=output_dir / "rank_ic_regime_confidence_intervals.png", title="Monthly Spearman IC by three-state VIX regime")
    print("[8/8] Writing formal inference research artifacts...")
    after_hashes = {path: _sha256(path) for path in protected_inputs}
    if before_hashes != after_hashes:
        raise AssertionError("Formal inference altered a protected descriptive input artifact.")
    (output_dir / "inference_summary.md").write_text(
        _summary_markdown(ls, lsd, ic, icd), encoding="utf-8"
    )
    metadata = pd.DataFrame([{
        "frequency": "monthly", "hac_estimator": "Newey-West/Bartlett HAC", "hac_lag": BASELINE_HAC_LAG,
        "confidence_level": CONFIDENCE_LEVEL, "pairwise_family_definition": "three contrasts per model/outcome/weighting where applicable",
        "primary_multiple_testing_method": "Holm", "secondary_multiple_testing_method": "Benjamini-Hochberg FDR",
        "forecasts_changed": False, "models_retrained": False, "portfolio_assignments_changed": False,
        "regime_definitions_changed": False,
        "long_short_sha256": before_hashes[source_dir / "long_short_returns.csv"],
        "rank_ic_sha256": before_hashes[source_dir / "rank_ic_monthly.csv"],
    }])
    metadata.to_csv(output_dir / "run_metadata.csv", index=False)
    return {
        "output_dir": output_dir,
        "long_short_inference": output_dir / "long_short_hac_inference.csv",
        "long_short_differences": output_dir / "long_short_regime_differences.csv",
        "rank_ic_inference": output_dir / "rank_ic_hac_inference.csv",
        "rank_ic_differences": output_dir / "rank_ic_regime_differences.csv",
        "summary": output_dir / "inference_summary.md",
    }

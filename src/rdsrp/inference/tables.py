"""Publication-table rendering for formal inference outputs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def write_long_short_latex(means: pd.DataFrame, differences: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{HAC inference for monthly $D10-D1$ returns across three-state VIX regimes}",
        r"\label{tab:long-short-inference}",
        r"\small",
        r"\begin{tabular}{lllrrrrrr}",
        r"\hline",
        r"Model & Weight & Regime & Mean & HAC SE & $t$ & $p$ & $N$ & Sig. \\",
        r"\hline",
    ]
    for row in means[means["regime_definition"] == "three_state"].itertuples(index=False):
        rows.append(
            f"{_escape(row.model)} & {_escape(row.weighting)} & {_escape(row.regime)} & "
            f"{row.mean_monthly_return:.5f} & {row.hac_se:.5f} & {row.t_stat:.2f} & "
            f"{row.p_value:.4f} & {int(row.n_months)} & {_escape(row.significance)} \\\\"
        )
    rows.extend([
        r"\hline",
        r"\multicolumn{9}{l}{\textit{Panel B: pairwise regime differences (Holm-adjusted inference)}} \\",
        r"Model & Weight & Comparison & Difference & HAC SE & $t$ & Raw $p$ & Holm $p$ & Sig. \\",
        r"\hline",
    ])
    for row in differences.itertuples(index=False):
        rows.append(
            f"{_escape(row.model)} & {_escape(row.weighting)} & {_escape(row.comparison)} & "
            f"{row.estimated_difference:.5f} & {row.hac_se:.5f} & {row.t_stat:.2f} & "
            f"{row.raw_p_value:.4f} & {row.holm_p_value:.4f} & {_escape(row.adjusted_significance)} \\\\"
        )
    rows.extend([
        r"\hline",
        r"\end{tabular}",
        r"\begin{flushleft}\footnotesize Notes: Newey--West/HAC covariance uses the repository's frozen six-month lag. Pairwise p-values are adjusted within model-by-weighting families using Holm; BH/FDR p-values remain in the machine-readable output.\end{flushleft}",
        r"\end{table}",
        "",
    ])
    path.write_text("\n".join(rows), encoding="utf-8")


def write_rank_ic_latex(means: pd.DataFrame, differences: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{HAC inference for monthly Spearman information coefficients}",
        r"\label{tab:rank-ic-three-state-inference}",
        r"\small",
        r"\begin{tabular}{llrrrrrr}",
        r"\hline",
        r"Model & Regime & Mean IC & HAC SE & $t$ & $p$ & $N$ & Sig. \\",
        r"\hline",
    ]
    for row in means[means["regime_definition"] == "three_state"].itertuples(index=False):
        rows.append(
            f"{_escape(row.model)} & {_escape(row.regime)} & {row.mean_spearman_ic:.5f} & "
            f"{row.hac_se:.5f} & {row.t_stat:.2f} & {row.p_value:.4f} & {int(row.n_months)} & "
            f"{_escape(row.significance)} \\\\"
        )
    rows.extend([
        r"\hline",
        r"\multicolumn{8}{l}{\textit{Panel B: pairwise regime differences (Holm-adjusted inference)}} \\",
        r"Model & Comparison & Difference & HAC SE & $t$ & Raw $p$ & Holm $p$ & Sig. \\",
        r"\hline",
    ])
    for row in differences.itertuples(index=False):
        rows.append(
            f"{_escape(row.model)} & {_escape(row.comparison)} & {row.estimated_difference:.5f} & "
            f"{row.hac_se:.5f} & {row.t_stat:.2f} & {row.raw_p_value:.4f} & "
            f"{row.holm_p_value:.4f} & {_escape(row.adjusted_significance)} \\\\"
        )
    rows.extend([
        r"\hline",
        r"\end{tabular}",
        r"\begin{flushleft}\footnotesize Notes: IC is the monthly cross-sectional Spearman correlation between predicted and realized next-month returns. HAC uses the frozen six-month Newey--West lag. Pairwise p-values are Holm-adjusted within model families.\end{flushleft}",
        r"\end{table}",
        "",
    ])
    path.write_text("\n".join(rows), encoding="utf-8")


def write_regime_differences_latex(
    long_short: pd.DataFrame, rank_ic: pd.DataFrame, path: Path
) -> None:
    """Write a compact two-panel table for three-state pairwise contrasts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Pairwise volatility-regime differences with HAC inference}",
        r"\label{tab:regime-difference-inference}",
        r"\small",
        r"\begin{tabular}{lllrrrrr}",
        r"\hline",
        r"\multicolumn{8}{l}{\textit{Panel A: $D10-D1$ returns}} \\",
        r"Model & Weight & Comparison & Difference & HAC SE & $t$ & Raw $p$ & Holm $p$ \\",
        r"\hline",
    ]
    for row in long_short.itertuples(index=False):
        rows.append(
            f"{_escape(row.model)} & {_escape(row.weighting)} & {_escape(row.comparison)} & "
            f"{row.estimated_difference:.5f} & {row.hac_se:.5f} & {row.t_stat:.2f} & "
            f"{row.raw_p_value:.4f} & {row.holm_p_value:.4f} \\\\"
        )
    rows.extend(
        [
            r"\hline",
            r"\multicolumn{8}{l}{\textit{Panel B: monthly Spearman IC}} \\",
            r"Model &  & Comparison & Difference & HAC SE & $t$ & Raw $p$ & Holm $p$ \\",
            r"\hline",
        ]
    )
    for row in rank_ic.itertuples(index=False):
        rows.append(
            f"{_escape(row.model)} &  & {_escape(row.comparison)} & "
            f"{row.estimated_difference:.5f} & {row.hac_se:.5f} & {row.t_stat:.2f} & "
            f"{row.raw_p_value:.4f} & {row.holm_p_value:.4f} \\\\"
        )
    rows.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\begin{flushleft}\footnotesize Notes: Differences are estimated as covariance-consistent linear contrasts from LOW-reference three-state regressions. Newey--West/HAC covariance uses six monthly lags. Holm adjustment is applied within each pre-specified three-comparison model/outcome/weighting family; BH/FDR p-values are retained in the CSV outputs.\end{flushleft}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(rows), encoding="utf-8")

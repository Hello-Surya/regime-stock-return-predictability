"""Multiple-testing adjustments for pre-specified regime-comparison families."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def add_multiple_testing_adjustments(
    frame: pd.DataFrame,
    *,
    family_cols: list[str],
    p_col: str = "raw_p_value",
) -> pd.DataFrame:
    """Add Holm FWER and Benjamini-Hochberg FDR p-values within each family."""
    if frame.empty:
        out = frame.copy()
        out["holm_p_value"] = pd.Series(dtype=float)
        out["bh_fdr_p_value"] = pd.Series(dtype=float)
        return out
    required = set(family_cols) | {p_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Multiple-testing table is missing columns: {missing}")
    out = frame.copy()
    out["holm_p_value"] = np.nan
    out["bh_fdr_p_value"] = np.nan
    grouped = out.groupby(family_cols, sort=False, dropna=False).groups
    for _, indices in grouped.items():
        loc = list(indices)
        pvals = pd.to_numeric(out.loc[loc, p_col], errors="coerce").to_numpy(float)
        if not np.isfinite(pvals).all():
            raise ValueError("All raw p-values must be finite before multiple-testing adjustment.")
        out.loc[loc, "holm_p_value"] = multipletests(pvals, method="holm")[1]
        out.loc[loc, "bh_fdr_p_value"] = multipletests(pvals, method="fdr_bh")[1]
    return out

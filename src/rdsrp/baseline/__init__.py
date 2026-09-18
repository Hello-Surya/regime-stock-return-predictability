"""Public API for baseline model estimation."""
import numpy as np
import pandas as pd

from .data import (
    BASELINE_PREDICTORS,
    BENCHMARK_DEFINITION,
    BENCHMARK_NAME,
    CANONICAL_PANEL_RELATIVE_PATH,
    EXPECTED_PRODUCTION_COUNTS,
    canonical_panel_path,
    prepare_baseline_sample as _prepare_baseline_sample,
    production_count_audit,
)
from .estimation import (
    deterministic_tuning_sample,
    run_expanding_baseline_oos,
    validate_prediction_panel,
)
from .quick import make_validation_panel, run_quick_baseline, run_software_validation
from .reporting import (
    baseline_model_metrics,
    monthly_rank_correlations,
    prediction_diagnostics,
    summarize_rank_metrics,
)
from .runner import run_production_baseline


def prepare_baseline_sample(panel: pd.DataFrame, *, min_price: float = 5.0) -> pd.DataFrame:
    """Public preparation wrapper retaining optional timing and regime audits."""
    formation = pd.to_datetime(panel["formation_date"]) + pd.offsets.MonthEnd(0)
    for column in (
        "accounting_available_date",
        "book_to_market_available_date",
        "bm_available_date",
        "accounting_assignment_date",
    ):
        if column in panel:
            available = pd.to_datetime(panel[column], errors="coerce") + pd.offsets.MonthEnd(0)
            bad = available.notna() & (available > formation)
            if bad.any():
                raise AssertionError(
                    f"Accounting timing violation: {int(bad.sum())} rows have {column} after formation."
                )
    sample = _prepare_baseline_sample(panel, min_price=min_price)
    if "vix_expanding_median" in sample:
        median = pd.to_numeric(sample["vix_expanding_median"], errors="coerce")
        implied = np.where(sample["vix"].to_numpy(float) > median.to_numpy(float), "HIGH", "LOW")
        valid = median.notna().to_numpy()
        observed = sample["regime"].astype(str).str.upper().to_numpy()
        if valid.any() and not np.array_equal(observed[valid], implied[valid]):
            raise AssertionError("Saved VIX regime labels do not match the frozen expanding-median rule.")
    return sample


__all__ = [
    "BASELINE_PREDICTORS",
    "BENCHMARK_DEFINITION",
    "BENCHMARK_NAME",
    "CANONICAL_PANEL_RELATIVE_PATH",
    "EXPECTED_PRODUCTION_COUNTS",
    "baseline_model_metrics",
    "canonical_panel_path",
    "deterministic_tuning_sample",
    "make_validation_panel",
    "monthly_rank_correlations",
    "prediction_diagnostics",
    "prepare_baseline_sample",
    "production_count_audit",
    "run_expanding_baseline_oos",
    "run_production_baseline",
    "run_quick_baseline",
    "run_software_validation",
    "summarize_rank_metrics",
    "validate_prediction_panel",
]

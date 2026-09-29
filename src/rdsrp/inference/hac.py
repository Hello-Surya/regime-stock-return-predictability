"""Reusable Newey-West/HAC inference for monthly research series."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASELINE_HAC_LAG = 6
CONFIDENCE_LEVEL = 0.95
REGIME_3STATE_ORDER = ("LOW", "MIDDLE", "HIGH")
BINARY_REGIME_ORDER = ("LOW", "HIGH")
PAIRWISE_COMPARISONS = (
    ("MIDDLE", "LOW"),
    ("HIGH", "LOW"),
    ("HIGH", "MIDDLE"),
)


@dataclass(frozen=True)
class HACResult:
    estimate: float
    hac_se: float
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    n_months: int
    hac_lag: int


def significance_stars(p_value: float) -> str:
    """Return conventional publication stars with strict p-value cutoffs."""
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def _critical_value(confidence_level: float) -> float:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one.")
    return float(NormalDist().inv_cdf(0.5 + confidence_level / 2.0))


def _result_from_linear_combination(
    fit: object,
    contrast: np.ndarray,
    *,
    n_months: int,
    hac_lag: int,
    confidence_level: float,
) -> HACResult:
    params = np.asarray(fit.params, dtype=float)
    covariance = np.asarray(fit.cov_params(), dtype=float)
    contrast = np.asarray(contrast, dtype=float)
    estimate = float(contrast @ params)
    variance = float(contrast @ covariance @ contrast)
    variance = max(variance, 0.0)
    se = float(np.sqrt(variance))
    t_stat = estimate / se if se > 0.0 else float("nan")
    p_value = float(2.0 * (1.0 - NormalDist().cdf(abs(t_stat)))) if np.isfinite(t_stat) else float("nan")
    z = _critical_value(confidence_level)
    return HACResult(
        estimate=estimate,
        hac_se=se,
        t_stat=t_stat,
        p_value=p_value,
        ci_lower=estimate - z * se,
        ci_upper=estimate + z * se,
        n_months=int(n_months),
        hac_lag=int(hac_lag),
    )


def hac_mean(
    values: pd.Series | np.ndarray,
    *,
    maxlags: int = BASELINE_HAC_LAG,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> HACResult:
    """Estimate a mean as an intercept-only OLS regression with HAC covariance."""
    y = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 2:
        raise ValueError("At least two finite monthly observations are required for HAC inference.")
    if maxlags < 0:
        raise ValueError("maxlags must be non-negative.")
    x = np.ones((len(y), 1), dtype=float)
    fit = sm.OLS(y, x).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": int(maxlags), "use_correction": False},
        use_t=False,
    )
    return _result_from_linear_combination(
        fit,
        np.array([1.0]),
        n_months=len(y),
        hac_lag=maxlags,
        confidence_level=confidence_level,
    )


def fit_regime_regression(
    frame: pd.DataFrame,
    *,
    outcome_col: str,
    regime_col: str,
    regime_order: tuple[str, ...] = REGIME_3STATE_ORDER,
    maxlags: int = BASELINE_HAC_LAG,
) -> tuple[object, pd.DataFrame]:
    """Fit LOW-reference regime dummies with HAC covariance in calendar order."""
    if len(regime_order) < 2:
        raise ValueError("regime_order must contain at least two regimes.")
    required = {outcome_col, regime_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Regime regression is missing columns: {missing}")
    work = frame[[outcome_col, regime_col]].copy()
    work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce")
    work[regime_col] = work[regime_col].astype(str).str.upper()
    work = work[np.isfinite(work[outcome_col]) & work[regime_col].isin(regime_order)]
    if work.empty:
        raise ValueError("Regime regression has no valid observations.")
    missing_regimes = [r for r in regime_order if not (work[regime_col] == r).any()]
    if missing_regimes:
        raise ValueError(f"Regime regression is missing required regimes: {missing_regimes}")

    x = pd.DataFrame({"const": 1.0}, index=work.index)
    for regime in regime_order[1:]:
        x[regime] = (work[regime_col] == regime).astype(float)
    fit = sm.OLS(work[outcome_col].to_numpy(float), x.to_numpy(float)).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": int(maxlags), "use_correction": False},
        use_t=False,
    )
    return fit, work


def regime_mean_from_fit(
    fit: object,
    regime: str,
    *,
    regime_order: tuple[str, ...] = REGIME_3STATE_ORDER,
    n_months: int,
    hac_lag: int = BASELINE_HAC_LAG,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> HACResult:
    regime = regime.upper()
    if regime not in regime_order:
        raise ValueError(f"Unknown regime: {regime}")
    contrast = np.zeros(len(regime_order), dtype=float)
    contrast[0] = 1.0
    if regime != regime_order[0]:
        contrast[regime_order.index(regime)] = 1.0
    return _result_from_linear_combination(
        fit,
        contrast,
        n_months=n_months,
        hac_lag=hac_lag,
        confidence_level=confidence_level,
    )


def regime_difference_from_fit(
    fit: object,
    left: str,
    right: str,
    *,
    regime_order: tuple[str, ...] = REGIME_3STATE_ORDER,
    n_months: int,
    hac_lag: int = BASELINE_HAC_LAG,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> HACResult:
    """Return the covariance-consistent HAC contrast mean(left)-mean(right)."""
    left = left.upper()
    right = right.upper()
    if left not in regime_order or right not in regime_order or left == right:
        raise ValueError("left/right must be distinct members of regime_order.")
    def mean_contrast(regime: str) -> np.ndarray:
        c = np.zeros(len(regime_order), dtype=float)
        c[0] = 1.0
        if regime != regime_order[0]:
            c[regime_order.index(regime)] = 1.0
        return c
    contrast = mean_contrast(left) - mean_contrast(right)
    return _result_from_linear_combination(
        fit,
        contrast,
        n_months=n_months,
        hac_lag=hac_lag,
        confidence_level=confidence_level,
    )

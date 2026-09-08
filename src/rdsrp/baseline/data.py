"""Production sample loading and validation for baseline estimation."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

BASELINE_PREDICTORS: tuple[str, ...] = ("log_me", "book_to_market", "mom_12_2")
EXPECTED_PRODUCTION_COUNTS: dict[str, int] = {
    "total_stock_months": 2_140_091,
    "complete_predictor_rows": 1_798_891,
    "final_screened_rows": 1_340_823,
    "duplicate_permno_date_rows": 0,
}
BENCHMARK_NAME = "pooled_expanding_historical_mean"
BENCHMARK_DEFINITION = (
    "At formation month t, every stock receives the pooled mean of all next-month returns "
    "whose targets are observable by t. The benchmark uses no observations dated after t."
)


def load_production_config(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "configs" / "production.yaml"
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")
    return data


def _month_end(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series) + pd.offsets.MonthEnd(0)


def _price_eligible(panel: pd.DataFrame, min_price: float) -> pd.Series:
    if "price_eligible" in panel:
        return panel["price_eligible"].fillna(False).astype(bool)
    if "prc" in panel:
        return pd.to_numeric(panel["prc"], errors="coerce").ge(min_price)
    raise ValueError("Production panel must contain price_eligible or prc for the frozen price screen.")


def _assert_optional_accounting_timing(panel: pd.DataFrame) -> None:
    """Reject saved rows if an explicit accounting availability field is after formation."""
    if "date" not in panel:
        return
    formation = _month_end(panel["date"])
    for column in (
        "accounting_available_date",
        "book_to_market_available_date",
        "bm_available_date",
        "accounting_assignment_date",
    ):
        if column not in panel:
            continue
        available = pd.to_datetime(panel[column], errors="coerce") + pd.offsets.MonthEnd(0)
        violation = available.notna() & (available > formation)
        if violation.any():
            raise AssertionError(
                f"Accounting timing violation: {int(violation.sum())} rows have {column} after formation."
            )


def production_count_audit(
    panel: pd.DataFrame,
    *,
    min_price: float = 5.0,
    expected: Mapping[str, int] | None = EXPECTED_PRODUCTION_COUNTS,
    strict: bool = True,
) -> dict[str, int]:
    required = {"date", "permno", "next_month_return", "regime", "vix", *BASELINE_PREDICTORS}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Completed production panel is missing required columns: {sorted(missing)}")
    complete = panel[list(BASELINE_PREDICTORS)].notna().all(axis=1)
    final = (
        complete
        & _price_eligible(panel, min_price)
        & panel["next_month_return"].notna()
        & panel["regime"].notna()
        & panel["vix"].notna()
    )
    audit = {
        "total_stock_months": int(len(panel)),
        "complete_predictor_rows": int(complete.sum()),
        "final_screened_rows": int(final.sum()),
        "duplicate_permno_date_rows": int(panel.duplicated(["permno", "date"]).sum()),
    }
    if strict and expected is not None:
        mismatches = {
            key: (audit.get(key), int(value))
            for key, value in expected.items()
            if audit.get(key) != int(value)
        }
        if mismatches:
            details = ", ".join(
                f"{key}: observed={obs:,} expected={exp:,}" for key, (obs, exp) in mismatches.items()
            )
            raise RuntimeError(
                "Completed production-panel counts differ from the validated snapshot; estimation stopped. "
                + details
            )
    return audit


def prepare_baseline_sample(panel: pd.DataFrame, *, min_price: float = 5.0) -> pd.DataFrame:
    """Return the frozen complete-case baseline sample without altering predictor values."""
    _assert_optional_accounting_timing(panel)
    required = {"date", "permno", "next_month_return", "regime", "vix", *BASELINE_PREDICTORS}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Production panel is missing required model fields: {sorted(missing)}")
    keep = (
        _price_eligible(panel, min_price)
        & panel[list(BASELINE_PREDICTORS)].notna().all(axis=1)
        & panel["next_month_return"].notna()
        & panel["regime"].notna()
        & panel["vix"].notna()
    )
    optional = [
        c
        for c in (
            "ticker",
            "prc",
            "price_eligible",
            "me_lag",
            "me_lag_date",
            "market_equity",
            "realized_return_date",
            "vix_expanding_median",
            "accounting_available_date",
            "book_to_market_available_date",
            "bm_available_date",
            "accounting_assignment_date",
        )
        if c in panel.columns
    ]
    columns = list(
        dict.fromkeys(
            ["date", "permno", "next_month_return", "regime", "vix", *BASELINE_PREDICTORS, *optional]
        )
    )
    sample = panel.loc[keep, columns].copy()
    sample["date"] = _month_end(sample["date"])
    if "realized_return_date" in sample:
        sample["realized_return_date"] = _month_end(sample["realized_return_date"])
    else:
        sample["realized_return_date"] = sample["date"] + pd.offsets.MonthEnd(1)
    expected_realized = sample["date"] + pd.offsets.MonthEnd(1)
    if not sample["realized_return_date"].eq(expected_realized).all():
        raise AssertionError(
            "Target timing is not calendar-adjacent: next_month_return must be realized in t+1."
        )
    if sample.duplicated(["permno", "date"]).any():
        raise AssertionError("Baseline sample contains duplicate permno/date rows.")
    regimes = set(sample["regime"].dropna().astype(str).str.upper().unique())
    if not regimes.issubset({"HIGH", "LOW"}):
        raise ValueError(f"Unexpected volatility-regime labels: {sorted(regimes)}")
    if "vix_expanding_median" in sample.columns:
        median = pd.to_numeric(sample["vix_expanding_median"], errors="coerce")
        implied = np.where(sample["vix"].to_numpy(float) > median.to_numpy(float), "HIGH", "LOW")
        valid = median.notna().to_numpy()
        observed = sample["regime"].astype(str).str.upper().to_numpy()
        if valid.any() and not np.array_equal(observed[valid], implied[valid]):
            raise AssertionError("Saved VIX regime labels do not match the frozen expanding-median rule.")
    for col in BASELINE_PREDICTORS:
        sample[col] = pd.to_numeric(sample[col], errors="raise").astype("float32")
    sample["next_month_return"] = pd.to_numeric(
        sample["next_month_return"], errors="raise"
    ).astype("float32")
    sample["vix"] = pd.to_numeric(sample["vix"], errors="raise").astype("float32")
    return sample.sort_values(["date", "permno"], kind="mergesort").reset_index(drop=True)


def read_production_panel(panel_path: Path) -> pd.DataFrame:
    """Load only columns needed by the production estimator."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required to read the production Parquet panel.") from exc
    schema = set(pq.ParquetFile(panel_path).schema.names)
    mandatory = {"date", "permno", "next_month_return", "regime", "vix", *BASELINE_PREDICTORS}
    if not mandatory.issubset(schema):
        raise ValueError(f"Production Parquet is missing columns: {sorted(mandatory.difference(schema))}")
    desired = list(mandatory)
    for optional in (
        "ticker",
        "prc",
        "price_eligible",
        "me_lag",
        "me_lag_date",
        "market_equity",
        "realized_return_date",
        "vix_expanding_median",
        "accounting_available_date",
        "book_to_market_available_date",
        "bm_available_date",
        "accounting_assignment_date",
    ):
        if optional in schema:
            desired.append(optional)
    return pd.read_parquet(panel_path, columns=list(dict.fromkeys(desired)))

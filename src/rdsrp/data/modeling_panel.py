"""Construction and validation helpers for the complete baseline predictor panel."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rdsrp.features.book_equity import (
    assign_book_to_market_monthly,
    construct_annual_book_to_market,
    link_accounting_to_permno,
    prepare_annual_accounting,
)
from rdsrp.features.build import build_features

PREDICTORS: tuple[str, ...] = ("log_me", "book_to_market", "mom_12_2")
PERCENTILES: tuple[float, ...] = (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)


def build_complete_predictor_panel(
    crsp: pd.DataFrame,
    compustat: pd.DataFrame,
    ccm: pd.DataFrame,
    regimes: pd.DataFrame,
    *,
    min_price: float = 5.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the size/book-to-market/momentum panel without training models."""
    if crsp.duplicated(["permno", "date"]).any():
        raise ValueError("Raw CRSP input contains duplicate PERMNO-date observations.")

    features = build_features(crsp)
    annual, annual_stats = prepare_annual_accounting(compustat)
    linked, link_stats = link_accounting_to_permno(annual, ccm)
    annual_bm, bm_stats = construct_annual_book_to_market(linked, features)
    panel, merge_stats = assign_book_to_market_monthly(features, annual_bm)

    regime_cols = [
        c for c in ["date", "vix", "vix_expanding_median", "regime"] if c in regimes.columns
    ]
    if "date" not in regime_cols:
        raise ValueError("Regime data must contain date.")
    r = regimes[regime_cols].copy()
    r["date"] = pd.to_datetime(r["date"]) + pd.offsets.MonthEnd(0)
    if r.duplicated(["date"]).any():
        raise ValueError("Regime input contains duplicate monthly dates.")
    panel = panel.merge(r, on="date", how="left", validate="many_to_one")

    # Keep CRSP's established $-thousands ME convention in the core feature panel.
    # ``me_dec`` is the separately normalized $-millions denominator used for B/M.
    panel["me"] = pd.to_numeric(panel["market_equity"], errors="coerce")
    panel["price_eligible"] = pd.to_numeric(panel["prc"], errors="coerce").ge(float(min_price))
    panel = panel.sort_values(["date", "permno"]).reset_index(drop=True)

    duplicates = int(panel.duplicated(["permno", "date"]).sum())
    if duplicates:
        raise RuntimeError(f"Final modeling panel has {duplicates} duplicate PERMNO-date rows.")

    audit: dict[str, Any] = {}
    audit.update(annual_stats)
    audit.update(link_stats)
    audit.update(bm_stats)
    audit.update(merge_stats)
    audit.update(
        {
            "raw_crsp_rows": int(len(crsp)),
            "final_panel_rows": int(len(panel)),
            "unique_permnos": int(panel["permno"].nunique()),
            "duplicate_permno_date_rows": duplicates,
            "valid_log_me_rows": int(panel["log_me"].notna().sum()),
            "valid_momentum_rows": int(panel["mom_12_2"].notna().sum()),
            "valid_book_to_market_months": int(panel["book_to_market"].notna().sum()),
            "complete_predictor_rows": int(panel[list(PREDICTORS)].notna().all(axis=1).sum()),
        }
    )
    return panel, audit


def _summary_row(name: str, series: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(series, errors="coerce").dropna()
    row: dict[str, Any] = {
        "variable": name,
        "n": int(len(x)),
        "missing": int(series.isna().sum()),
        "min": float(x.min()) if len(x) else np.nan,
        "max": float(x.max()) if len(x) else np.nan,
    }
    for q in PERCENTILES:
        row[f"p{q * 100:g}"] = float(x.quantile(q)) if len(x) else np.nan
    return row


def distribution_report(panel: pd.DataFrame) -> pd.DataFrame:
    fields = ["book_equity", "me_dec", "book_to_market", "log_me", "mom_12_2"]
    return pd.DataFrame([_summary_row(c, panel[c]) for c in fields if c in panel.columns])


def sample_flow(panel: pd.DataFrame, *, min_price: float = 5.0) -> pd.DataFrame:
    """Create an auditable sequential stock-month sample-flow table."""
    x = panel.copy()
    missing_series = pd.Series(np.nan, index=x.index)
    conditions: list[tuple[str, pd.Series]] = [
        ("Raw CRSP eligible observations", pd.Series(True, index=x.index)),
        ("Valid next-month target", x["next_month_return"].notna()),
        (f"Price >= ${min_price:g}", pd.to_numeric(x["prc"], errors="coerce").ge(min_price)),
        ("Valid size", x["log_me"].notna()),
        ("Valid momentum", x["mom_12_2"].notna()),
        ("Matched CCM", x.get("gvkey", missing_series).notna()),
        ("Valid book equity", pd.to_numeric(x.get("book_equity", missing_series), errors="coerce").gt(0)),
        ("Valid book-to-market", x.get("book_to_market", missing_series).notna()),
    ]
    keep = pd.Series(True, index=x.index)
    rows: list[dict[str, Any]] = []
    for stage, cond in conditions:
        keep &= cond.fillna(False)
        rows.append({"stage": stage, "stock_month_observations": int(keep.sum())})
    complete = keep & x[list(PREDICTORS)].notna().all(axis=1)
    rows.append(
        {
            "stage": "Complete baseline predictor set",
            "stock_month_observations": int(complete.sum()),
        }
    )
    regime = x.get("regime", pd.Series(pd.NA, index=x.index)).notna()
    rows.append(
        {
            "stage": "Complete predictors + target + regime",
            "stock_month_observations": int(
                (complete & x["next_month_return"].notna() & regime).sum()
            ),
        }
    )
    return pd.DataFrame(rows)


def validate_economic_sanity(panel: pd.DataFrame) -> dict[str, Any]:
    """Return transparent unit/arithmetic checks; fail only on structural errors."""
    result: dict[str, Any] = {
        "duplicate_permno_date_rows": int(panel.duplicated(["permno", "date"]).sum()),
        "market_equity_identity_max_abs_error": None,
        "book_to_market_identity_max_abs_error": None,
        "book_to_market_median": None,
        "book_to_market_p999": None,
        "unit_sanity_status": "not_evaluable",
    }
    if result["duplicate_permno_date_rows"]:
        raise RuntimeError("Duplicate PERMNO-date rows detected during economic sanity checks.")

    required_me = {"market_equity", "prc", "shrout"}
    if required_me.issubset(panel.columns):
        lhs = pd.to_numeric(panel["market_equity"], errors="coerce")
        rhs = pd.to_numeric(panel["prc"], errors="coerce").abs() * pd.to_numeric(
            panel["shrout"], errors="coerce"
        )
        error = (lhs - rhs).abs().dropna()
        result["market_equity_identity_max_abs_error"] = float(error.max()) if len(error) else None

    if {"book_equity", "me_dec", "book_to_market"}.issubset(panel.columns):
        valid = (
            pd.to_numeric(panel["book_equity"], errors="coerce").gt(0)
            & pd.to_numeric(panel["me_dec"], errors="coerce").gt(0)
            & panel["book_to_market"].notna()
        )
        expected = pd.to_numeric(panel.loc[valid, "book_equity"], errors="coerce") / pd.to_numeric(
            panel.loc[valid, "me_dec"], errors="coerce"
        )
        actual = pd.to_numeric(panel.loc[valid, "book_to_market"], errors="coerce")
        bm_error = (actual - expected).abs().dropna()
        result["book_to_market_identity_max_abs_error"] = (
            float(bm_error.max()) if len(bm_error) else None
        )

    bm = pd.to_numeric(panel.get("book_to_market"), errors="coerce").dropna()
    if len(bm):
        median = float(bm.median())
        p999 = float(bm.quantile(0.999))
        result["book_to_market_median"] = median
        result["book_to_market_p999"] = p999
        # These broad bounds are diagnostic guards against scale mistakes, not winsorization.
        if 0.001 <= median <= 50.0:
            result["unit_sanity_status"] = "pass"
        else:
            result["unit_sanity_status"] = "fail_scale_check"
            raise RuntimeError(
                "Book-to-market median is outside the broad unit-sanity range; inspect source units "
                "and merges before continuing."
            )
    return result


def construction_summary(
    compustat: pd.DataFrame,
    ccm: pd.DataFrame,
    panel: pd.DataFrame,
    audit: dict[str, Any],
) -> dict[str, Any]:
    missing = pd.Series(np.nan, index=panel.index)
    be = pd.to_numeric(panel.get("book_equity", missing), errors="coerce")
    bm = pd.to_numeric(panel.get("book_to_market", missing), errors="coerce")
    total = len(panel)
    if total and "gvkey" in panel.columns:
        security_match = panel.assign(_matched=panel["gvkey"].notna()).groupby("permno")["_matched"].any()
        unmatched_crsp_securities = int((~security_match).sum())
    else:
        unmatched_crsp_securities = int(panel["permno"].nunique()) if total else 0
    accounting_rows = int(audit.get("accounting_rows", 0))
    accounting_matched = int(audit.get("accounting_rows_with_valid_ccm", 0))
    ccm_match_rate = float(accounting_matched / accounting_rows) if accounting_rows else 0.0

    def coverage(column: str) -> float:
        if total == 0 or column not in panel:
            return 0.0
        return float(panel[column].notna().mean())

    return {
        # Preserve construction-stage diagnostics, but let research-sample metrics below
        # take precedence when a key exists in both dictionaries.  The audit is
        # computed on the buffered construction panel, while ``panel`` may be the
        # narrower configured research sample.
        **audit,
        "compustat_rows": int(len(compustat)),
        "compustat_unique_gvkeys": int(compustat["gvkey"].nunique()) if "gvkey" in compustat else 0,
        "compustat_fiscal_start": (
            str(pd.to_datetime(compustat["datadate"]).min().date()) if len(compustat) else None
        ),
        "compustat_fiscal_end": (
            str(pd.to_datetime(compustat["datadate"]).max().date()) if len(compustat) else None
        ),
        "compustat_missing_seq": int(compustat["seq"].isna().sum()) if "seq" in compustat else None,
        "compustat_missing_ceq": int(compustat["ceq"].isna().sum()) if "ceq" in compustat else None,
        "compustat_missing_at": int(compustat["at"].isna().sum()) if "at" in compustat else None,
        "compustat_missing_lt": int(compustat["lt"].isna().sum()) if "lt" in compustat else None,
        "ccm_rows": int(len(ccm)),
        "ccm_accounting_match_rate": ccm_match_rate,
        "unmatched_crsp_securities": unmatched_crsp_securities,
        "ccm_link_type_distribution": (
            {str(k): int(v) for k, v in ccm["linktype"].astype("string").value_counts(dropna=False).items()}
            if "linktype" in ccm
            else {}
        ),
        "ccm_link_primary_distribution": (
            {str(k): int(v) for k, v in ccm["linkprim"].astype("string").value_counts(dropna=False).items()}
            if "linkprim" in ccm
            else {}
        ),
        "panel_rows": int(total),
        "panel_unique_permnos": int(panel["permno"].nunique()),
        "panel_start": str(pd.to_datetime(panel["date"]).min().date()) if total else None,
        "panel_end": str(pd.to_datetime(panel["date"]).max().date()) if total else None,
        "book_equity_missing_months": int(be.isna().sum()),
        "book_equity_zero_months": int(be.eq(0).sum()),
        "book_equity_negative_months": int(be.lt(0).sum()),
        "valid_book_to_market_months": int(bm.notna().sum()),
        "coverage_log_me": coverage("log_me"),
        "coverage_mom_12_2": coverage("mom_12_2"),
        "coverage_book_to_market": coverage("book_to_market"),
        "coverage_next_month_return": coverage("next_month_return"),
        "coverage_regime": coverage("regime"),
        "complete_predictor_rows": int(panel[list(PREDICTORS)].notna().all(axis=1).sum()),
        "duplicate_permno_date_rows": int(panel.duplicated(["permno", "date"]).sum()),
    }

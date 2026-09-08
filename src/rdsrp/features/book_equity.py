"""Book-equity construction and leakage-safe annual book-to-market assignment."""
from __future__ import annotations

import numpy as np
import pandas as pd

VALID_CCM_LINK_TYPES: tuple[str, ...] = ("LC", "LU")
VALID_CCM_LINK_PRIMARY: tuple[str, ...] = ("P", "C")
CRSP_ME_THOUSANDS_TO_MILLIONS = 1_000.0


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def construct_stockholders_equity(funda: pd.DataFrame) -> pd.DataFrame:
    out = funda.copy()
    seq, ceq, pstk = _numeric(out, "seq"), _numeric(out, "ceq"), _numeric(out, "pstk")
    at, lt = _numeric(out, "at"), _numeric(out, "lt")
    she = seq.copy()
    source = pd.Series(pd.NA, index=out.index, dtype="string")
    source.loc[seq.notna()] = "seq"
    use_ceq = she.isna() & ceq.notna() & pstk.notna()
    she.loc[use_ceq] = ceq.loc[use_ceq] + pstk.loc[use_ceq]
    source.loc[use_ceq] = "ceq_plus_pstk"
    use_assets = she.isna() & at.notna() & lt.notna()
    she.loc[use_assets] = at.loc[use_assets] - lt.loc[use_assets]
    source.loc[use_assets] = "at_minus_lt"
    out["stockholders_equity"] = she
    out["stockholders_equity_source"] = source
    return out


def construct_preferred_stock(funda: pd.DataFrame) -> pd.DataFrame:
    out = funda.copy()
    rv, lv, carrying = _numeric(out, "pstkrv"), _numeric(out, "pstkl"), _numeric(out, "pstk")
    preferred = rv.copy()
    source = pd.Series(pd.NA, index=out.index, dtype="string")
    source.loc[rv.notna()] = "pstkrv"
    use_lv = preferred.isna() & lv.notna()
    preferred.loc[use_lv] = lv.loc[use_lv]
    source.loc[use_lv] = "pstkl"
    use_carry = preferred.isna() & carrying.notna()
    preferred.loc[use_carry] = carrying.loc[use_carry]
    source.loc[use_carry] = "pstk"
    # Standard asset-pricing implementations treat an absent preferred-stock item as zero.
    missing = preferred.isna()
    preferred.loc[missing] = 0.0
    source.loc[missing] = "assumed_zero_missing"
    out["preferred_stock"] = preferred
    out["preferred_stock_source"] = source
    return out


def construct_deferred_taxes(funda: pd.DataFrame) -> pd.DataFrame:
    out = funda.copy()
    txditc, txdb, itcb = _numeric(out, "txditc"), _numeric(out, "txdb"), _numeric(out, "itcb")
    deferred = txditc.copy()
    source = pd.Series(pd.NA, index=out.index, dtype="string")
    source.loc[txditc.notna()] = "txditc"
    fallback = deferred.isna() & (txdb.notna() | itcb.notna())
    deferred.loc[fallback] = txdb.fillna(0.0).loc[fallback] + itcb.fillna(0.0).loc[fallback]
    source.loc[fallback] = "txdb_plus_itcb"
    missing = deferred.isna()
    # Absence of a deferred-tax / ITC item is explicitly treated as zero in the baseline.
    deferred.loc[missing] = 0.0
    source.loc[missing] = "assumed_zero_missing"
    out["deferred_taxes_itc"] = deferred
    out["deferred_taxes_source"] = source
    return out


def construct_book_equity(funda: pd.DataFrame) -> pd.DataFrame:
    out = construct_stockholders_equity(funda)
    out = construct_preferred_stock(out)
    out = construct_deferred_taxes(out)
    out["book_equity"] = (
        pd.to_numeric(out["stockholders_equity"], errors="coerce")
        + pd.to_numeric(out["deferred_taxes_itc"], errors="coerce")
        - pd.to_numeric(out["preferred_stock"], errors="coerce")
    )
    out["book_equity_valid"] = out["book_equity"].gt(0)
    return out


def prepare_annual_accounting(funda: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep one annual record per GVKEY/calendar fiscal-year-end year.

    For the June convention, a fiscal-year end in calendar year y-1 forms the
    characteristic for June y through May y+1. If multiple standard annual
    records exist in one calendar year, the latest fiscal-year-end record wins.
    """
    out = construct_book_equity(funda)
    out["gvkey"] = out["gvkey"].astype("string")
    out["datadate"] = pd.to_datetime(out["datadate"], errors="coerce")
    out = out.dropna(subset=["gvkey", "datadate"]).copy()
    out["accounting_year"] = out["datadate"].dt.year.astype("Int64")
    sort_cols = ["gvkey", "accounting_year", "datadate"] + (["fyear"] if "fyear" in out.columns else [])
    out = out.sort_values(sort_cols, na_position="first")
    duplicate_rows = int(out.duplicated(["gvkey", "accounting_year"], keep=False).sum())
    before = len(out)
    out = out.drop_duplicates(["gvkey", "accounting_year"], keep="last").copy()
    out["characteristic_year"] = out["accounting_year"] + 1
    out["accounting_available_from"] = pd.to_datetime(
        out["characteristic_year"].astype("string") + "-06-30", errors="coerce"
    )
    be = pd.to_numeric(out["book_equity"], errors="coerce")
    return out, {
        "annual_rows_input": int(before),
        "duplicate_firm_year_rows": duplicate_rows,
        "duplicate_firm_year_rows_removed": int(before - len(out)),
        "annual_book_equity_valid": int(be.gt(0).sum()),
        "annual_book_equity_missing": int(be.isna().sum()),
        "annual_book_equity_zero": int(be.eq(0).sum()),
        "annual_book_equity_negative": int(be.lt(0).sum()),
    }


def filter_ccm_links(ccm: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    out = ccm.copy()
    out["gvkey"] = out["gvkey"].astype("string")
    out["lpermno"] = pd.to_numeric(out["lpermno"], errors="coerce").astype("Int64")
    out["linktype"] = out["linktype"].astype("string").str.upper()
    out["linkprim"] = out["linkprim"].astype("string").str.upper()
    out["linkdt"] = pd.to_datetime(out["linkdt"], errors="coerce")
    out["linkenddt"] = pd.to_datetime(out["linkenddt"], errors="coerce")
    before = len(out)
    out = out[
        out["linktype"].isin(VALID_CCM_LINK_TYPES)
        & out["linkprim"].isin(VALID_CCM_LINK_PRIMARY)
        & out["gvkey"].notna()
        & out["lpermno"].notna()
    ].copy()
    return out, {"ccm_rows_input": int(before), "ccm_rows_retained": int(len(out))}


def link_accounting_to_permno(
    annual: pd.DataFrame,
    ccm: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    links, stats = filter_ccm_links(ccm)
    merged = annual.merge(links, on="gvkey", how="left", validate="many_to_many", suffixes=("", "_link"))
    valid_start = merged["linkdt"].isna() | (merged["linkdt"] <= merged["datadate"])
    valid_end = merged["linkenddt"].isna() | (merged["datadate"] <= merged["linkenddt"])
    valid = merged[valid_start & valid_end & merged["lpermno"].notna()].copy()
    valid["permno"] = valid["lpermno"].astype("Int64")
    # Resolve duplicate link rows for the same firm-year/security deterministically.
    valid["_linkprim_rank"] = valid["linkprim"].map({"P": 0, "C": 1}).fillna(9)
    valid["_linktype_rank"] = valid["linktype"].map({"LC": 0, "LU": 1}).fillna(9)
    valid = valid.sort_values(
        ["gvkey", "characteristic_year", "permno", "_linkprim_rank", "_linktype_rank", "linkdt"],
        ascending=[True, True, True, True, True, False],
        na_position="last",
    )
    duplicate_link_rows = int(valid.duplicated(["gvkey", "characteristic_year", "permno"], keep=False).sum())
    before_dedup = len(valid)
    valid = valid.drop_duplicates(["gvkey", "characteristic_year", "permno"], keep="first").copy()
    duplicate_resolved = int(before_dedup - len(valid))
    # A security can occasionally have competing GVKEY assignments. Resolve these
    # before firm-level market equity is formed: primary links first, then LC over
    # LU, latest accounting date, and finally GVKEY for deterministic tie-breaking.
    conflict_rows = int(valid.duplicated(["permno", "characteristic_year"], keep=False).sum())
    valid = valid.sort_values(
        ["permno", "characteristic_year", "_linkprim_rank", "_linktype_rank", "datadate", "gvkey"],
        ascending=[True, True, True, True, False, True],
        na_position="last",
    )
    before_conflict = len(valid)
    valid = valid.drop_duplicates(["permno", "characteristic_year"], keep="first").copy()
    valid = valid.drop(columns=["_linkprim_rank", "_linktype_rank"])
    matched_keys = valid[["gvkey", "characteristic_year"]].drop_duplicates()
    stats.update({
        "accounting_rows": int(len(annual)),
        "accounting_rows_with_valid_ccm": int(len(matched_keys)),
        "accounting_rows_unmatched_ccm": int(len(annual) - len(matched_keys)),
        "duplicate_ccm_rows_resolved": duplicate_resolved,
        "duplicate_ccm_rows_detected": duplicate_link_rows,
        "permno_year_gvkey_conflict_rows": conflict_rows,
        "permno_year_gvkey_conflicts_resolved": int(before_conflict - len(valid)),
    })
    return valid, stats


def build_december_market_equity(crsp_features: pd.DataFrame) -> pd.DataFrame:
    """Return one December market-equity observation per PERMNO/year.

    CRSP ``shrout`` is in thousands, so price*shrout is $ thousands. ``me_dec``
    is converted to $ millions for compatibility with Compustat fundamentals.
    """
    required = {"permno", "date", "market_equity"}
    missing = required.difference(crsp_features.columns)
    if missing:
        raise ValueError(f"CRSP features missing fields for December ME: {sorted(missing)}")
    dec = crsp_features.copy()
    dec["date"] = pd.to_datetime(dec["date"]) + pd.offsets.MonthEnd(0)
    dec = dec[dec["date"].dt.month == 12].copy()
    if dec.duplicated(["permno", "date"]).any():
        raise ValueError("Duplicate PERMNO-date rows found while constructing December market equity.")
    dec["december_year"] = dec["date"].dt.year.astype("Int64")
    dec["me_dec_security_thousands"] = pd.to_numeric(dec["market_equity"], errors="coerce")
    dec["me_dec_security"] = dec["me_dec_security_thousands"] / CRSP_ME_THOUSANDS_TO_MILLIONS
    return dec[["permno", "december_year", "me_dec_security", "date"]].rename(columns={"date": "me_dec_date"})


def construct_annual_book_to_market(
    linked_annual: pd.DataFrame,
    crsp_features: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    dec = build_december_market_equity(crsp_features)
    annual = linked_annual.copy()
    before_all_links = len(annual)
    eligible_permnos = set(pd.to_numeric(crsp_features["permno"], errors="coerce").dropna().astype(int))
    annual = annual[pd.to_numeric(annual["permno"], errors="coerce").isin(eligible_permnos)].copy()
    outside_universe = before_all_links - len(annual)
    annual["december_year"] = pd.to_numeric(annual["characteristic_year"], errors="coerce").astype("Int64") - 1
    before = len(annual)
    annual = annual.merge(dec, on=["permno", "december_year"], how="left", validate="many_to_one")
    # Firm-level denominator: sum December ME across all valid CCM-linked CRSP
    # securities represented in the research universe. The firm-level BE numerator
    # is not divided by each share class separately.
    classes = annual.drop_duplicates(["gvkey", "characteristic_year", "permno"]).copy()
    class_counts = (
        classes.groupby(["gvkey", "characteristic_year"], as_index=False)
        .agg(
            linked_permnos=("permno", "nunique"),
            permnos_with_december_me=("me_dec_security", lambda x: int(x.notna().sum())),
            me_dec=("me_dec_security", lambda x: x.sum(min_count=1)),
        )
    )
    # If a linked security represented in the panel lacks its required December ME,
    # the firm denominator is incomplete and B/M is left missing rather than guessed.
    incomplete = class_counts["permnos_with_december_me"] < class_counts["linked_permnos"]
    class_counts.loc[incomplete, "me_dec"] = np.nan
    annual = annual.merge(
        class_counts,
        on=["gvkey", "characteristic_year"],
        how="left",
        validate="many_to_one",
    )
    book_equity = pd.to_numeric(annual["book_equity"], errors="coerce").astype("Float64")
    me_dec = pd.to_numeric(annual["me_dec"], errors="coerce").astype("Float64")
    valid_bm = (book_equity.gt(0) & me_dec.gt(0)).fillna(False)
    annual["book_to_market"] = pd.Series(np.nan, index=annual.index, dtype="float64")
    annual.loc[valid_bm, "book_to_market"] = (
        book_equity.loc[valid_bm].astype(float) / me_dec.loc[valid_bm].astype(float)
    )
    stats = {
        "linked_annual_rows_before_crsp_universe_filter": int(before_all_links),
        "linked_annual_rows_outside_crsp_universe": int(outside_universe),
        "linked_annual_rows": int(before),
        "linked_rows_with_december_me": int(annual["me_dec_security"].notna().sum()),
        "gvkey_years_multiple_permnos": int((class_counts["linked_permnos"] > 1).sum()),
        "max_permnos_per_gvkey_year": int(class_counts["linked_permnos"].max()) if len(class_counts) else 0,
        "gvkey_years_incomplete_december_me": int(incomplete.sum()),
        "valid_book_to_market_rows": int(annual["book_to_market"].notna().sum()),
    }
    return annual, stats


def assign_book_to_market_monthly(
    monthly: pd.DataFrame,
    annual_bm: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = monthly.copy()
    out["date"] = pd.to_datetime(out["date"]) + pd.offsets.MonthEnd(0)
    if out.duplicated(["permno", "date"]).any():
        raise ValueError("Monthly panel must have a unique PERMNO-date key before accounting merge.")
    out["characteristic_year"] = np.where(out["date"].dt.month >= 6, out["date"].dt.year, out["date"].dt.year - 1)
    out["characteristic_year"] = pd.Series(out["characteristic_year"], index=out.index, dtype="Int64")
    keep = [
        "permno", "characteristic_year", "gvkey", "datadate", "fyear", "fyr",
        "book_equity", "accounting_available_from", "me_dec", "book_to_market",
        "stockholders_equity_source", "preferred_stock_source", "deferred_taxes_source",
        "linkprim", "linktype",
    ]
    annual = annual_bm[[c for c in keep if c in annual_bm.columns]].copy()
    # A PERMNO can have at most one GVKEY assignment per characteristic year in the final panel.
    annual["_linkprim_rank"] = annual.get("linkprim", pd.Series("P", index=annual.index)).map({"P": 0, "C": 1}).fillna(9)
    annual = annual.sort_values(["permno", "characteristic_year", "_linkprim_rank", "datadate"], ascending=[True, True, True, False])
    duplicate_assignments = int(annual.duplicated(["permno", "characteristic_year"], keep=False).sum())
    annual = annual.drop_duplicates(["permno", "characteristic_year"], keep="first").drop(columns="_linkprim_rank")
    if "datadate" in annual.columns:
        annual = annual.rename(columns={"datadate": "accounting_datadate"})
    merged = out.merge(annual, on=["permno", "characteristic_year"], how="left", validate="many_to_one")
    if merged.duplicated(["permno", "date"]).any():
        raise RuntimeError("Accounting merge created duplicate PERMNO-date observations.")
    # Hard timing guard: even a malformed annual record cannot enter before its June availability date.
    early = merged["accounting_available_from"].notna() & (merged["date"] < merged["accounting_available_from"])
    if early.any():
        raise RuntimeError("Accounting information entered the monthly panel before its allowed June date.")
    return merged, {
        "monthly_rows_pre_merge": int(len(out)),
        "monthly_rows_post_merge": int(len(merged)),
        "duplicate_permno_characteristic_year_assignments_detected": duplicate_assignments,
        "duplicate_permno_date_rows": int(merged.duplicated(["permno", "date"]).sum()),
        "monthly_rows_with_book_to_market": int(merged["book_to_market"].notna().sum()),
    }

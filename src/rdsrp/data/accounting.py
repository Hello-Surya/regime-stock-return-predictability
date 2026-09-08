"""WRDS Compustat fundamentals and CRSP/Compustat Merged (CCM) utilities.

The source resolver inspects the connected WRDS account instead of assuming a
single historical schema. Raw licensed observations are cached locally by the
calling script and are never intended for version control.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from rdsrp.data.cache import read_parquet_cache, write_parquet_cache


@dataclass(frozen=True)
class CompustatSource:
    library: str
    table: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class CcmSource:
    library: str
    table: str
    columns: tuple[str, ...]


COMP_LIBRARY_PRIORITY: tuple[str, ...] = (
    "comp",
    "comp_na_daily_all",
    "comp_na_annual_all",
)
CCM_LIBRARY_PRIORITY: tuple[str, ...] = (
    "crsp",
    "crsp_a_ccm",
    "crsp_q_ccm",
)
COMP_TABLE_PRIORITY: tuple[str, ...] = ("funda",)
CCM_TABLE_PRIORITY: tuple[str, ...] = ("ccmxpf_lnkhist", "ccmxpf_linktable")
COMP_FIELDS: tuple[str, ...] = (
    "gvkey",
    "datadate",
    "fyear",
    "fyr",
    "seq",
    "ceq",
    "at",
    "lt",
    "txditc",
    "txdb",
    "itcb",
    "pstkrv",
    "pstkl",
    "pstk",
    "indfmt",
    "datafmt",
    "popsrc",
    "consol",
    "curcd",
    "fic",
)
CCM_FIELDS: tuple[str, ...] = (
    "gvkey",
    "lpermno",
    "permno",
    "linktype",
    "linkprim",
    "linkdt",
    "linkenddt",
)


def _lower_set(values: Iterable[str]) -> set[str]:
    return {str(value).lower() for value in values}


def _table_columns(db: Any, library: str, table: str) -> tuple[str, ...]:
    """Return lower-case table columns using WRDS metadata only."""
    desc = db.describe_table(library=library, table=table)
    if isinstance(desc, pd.DataFrame):
        lower_names = {str(c).lower(): c for c in desc.columns}
        for candidate in ("name", "column_name", "variable", "column"):
            if candidate in lower_names:
                source_col = lower_names[candidate]
                return tuple(str(x).lower() for x in desc[source_col].dropna())
        if desc.index.dtype == object:
            return tuple(str(x).lower() for x in desc.index)
    raise RuntimeError(f"Could not read column metadata for {library}.{table}.")


def _candidate_libraries(db: Any, priority: tuple[str, ...], token: str) -> list[str]:
    libraries = sorted(_lower_set(db.list_libraries()))
    ordered = [lib for lib in priority if lib in libraries]
    ordered.extend(lib for lib in libraries if lib not in ordered and token in lib)
    return ordered


def resolve_compustat_source(db: Any) -> CompustatSource:
    inspected: list[str] = []
    for library in _candidate_libraries(db, COMP_LIBRARY_PRIORITY, "comp"):
        try:
            tables = _lower_set(db.list_tables(library=library))
        except Exception:
            continue
        inspected.append(f"{library}({len(tables)} tables)")
        for table in COMP_TABLE_PRIORITY:
            if table not in tables:
                continue
            columns = _table_columns(db, library, table)
            required = {"gvkey", "datadate"}
            if required.issubset(columns) and ({"seq", "ceq", "at"} & set(columns)):
                return CompustatSource(library, table, columns)
    detail = ", ".join(inspected) or "no Compustat-like library accessible"
    raise RuntimeError(
        "Could not locate a Compustat annual fundamentals table with the required fields. "
        f"Inspected: {detail}."
    )


def resolve_ccm_source(db: Any) -> CcmSource:
    inspected: list[str] = []
    libraries = _candidate_libraries(db, CCM_LIBRARY_PRIORITY, "ccm")
    # Some accounts expose CCM inside a CRSP library whose name does not contain "ccm".
    all_libraries = sorted(_lower_set(db.list_libraries()))
    libraries.extend(lib for lib in all_libraries if lib.startswith("crsp") and lib not in libraries)
    for library in libraries:
        try:
            tables = _lower_set(db.list_tables(library=library))
        except Exception:
            continue
        inspected.append(f"{library}({len(tables)} tables)")
        for table in CCM_TABLE_PRIORITY:
            if table not in tables:
                continue
            columns = _table_columns(db, library, table)
            colset = set(columns)
            required = {"gvkey", "linktype", "linkprim", "linkdt", "linkenddt"}
            if required.issubset(colset) and ({"lpermno", "permno"} & colset):
                return CcmSource(library, table, columns)
    detail = ", ".join(inspected) or "no CCM-like library accessible"
    raise RuntimeError(
        "Could not locate a CCM link-history table with GVKEY, PERMNO and effective-date fields. "
        f"Inspected: {detail}."
    )


def _date_literal(value: str | pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _gvkey_filter(gvkeys: Sequence[str] | None) -> str | None:
    if not gvkeys:
        return None
    cleaned = sorted({str(value).strip() for value in gvkeys if str(value).strip()})
    if not cleaned:
        return None
    quoted = ", ".join("'" + value.replace("'", "''") + "'" for value in cleaned)
    return f"gvkey IN ({quoted})"


def build_compustat_query(
    source: CompustatSource,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    *,
    gvkeys: Sequence[str] | None = None,
    row_limit: int | None = None,
) -> str:
    cols = [c for c in COMP_FIELDS if c in source.columns]
    if not {"gvkey", "datadate"}.issubset(cols):
        raise ValueError("Resolved Compustat source lacks GVKEY or DATADATE.")
    filters = [f"datadate BETWEEN '{_date_literal(start_date)}' AND '{_date_literal(end_date)}'"]
    # Standard annual industrial, consolidated, domestic-population data.
    standard_filters = {
        "indfmt": "INDL",
        "datafmt": "STD",
        "popsrc": "D",
        "consol": "C",
        "curcd": "USD",
    }
    for col, value in standard_filters.items():
        if col in source.columns:
            filters.append(f"{col} = '{value}'")
    gvkey_clause = _gvkey_filter(gvkeys)
    if gvkey_clause is not None:
        filters.append(gvkey_clause)
    limit = f"\nLIMIT {int(row_limit)}" if row_limit is not None else ""
    return (
        "SELECT\n    "
        + ",\n    ".join(cols)
        + f"\nFROM {source.library}.{source.table}\nWHERE "
        + "\n  AND ".join(filters)
        + f"\nORDER BY gvkey, datadate{limit}"
    )


def pull_compustat_fundamentals(
    db: Any,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    *,
    source: CompustatSource | None = None,
    gvkeys: Sequence[str] | None = None,
    row_limit: int | None = None,
) -> tuple[pd.DataFrame, CompustatSource]:
    resolved = source or resolve_compustat_source(db)
    raw = db.raw_sql(
        build_compustat_query(
            resolved,
            start_date,
            end_date,
            gvkeys=gvkeys,
            row_limit=row_limit,
        ),
        date_cols=["datadate"],
    )
    out = raw.copy()
    out.columns = [str(c).lower() for c in out.columns]
    for col in COMP_FIELDS:
        if col not in out.columns:
            out[col] = pd.NA
    out["gvkey"] = out["gvkey"].astype("string")
    out["datadate"] = pd.to_datetime(out["datadate"], errors="coerce")
    out["source_table"] = f"{resolved.library}.{resolved.table}"
    return out, resolved


def build_ccm_query(source: CcmSource, *, row_limit: int | None = None) -> str:
    permno_col = "lpermno" if "lpermno" in source.columns else "permno"
    cols = ["gvkey", f"{permno_col} AS lpermno", "linktype", "linkprim", "linkdt", "linkenddt"]
    limit = f"\nLIMIT {int(row_limit)}" if row_limit is not None else ""
    return (
        "SELECT\n    "
        + ",\n    ".join(cols)
        + f"\nFROM {source.library}.{source.table}\n"
        + f"ORDER BY gvkey, linkdt, lpermno{limit}"
    )


def pull_ccm_links(
    db: Any,
    *,
    source: CcmSource | None = None,
    row_limit: int | None = None,
) -> tuple[pd.DataFrame, CcmSource]:
    resolved = source or resolve_ccm_source(db)
    raw = db.raw_sql(
        build_ccm_query(resolved, row_limit=row_limit),
        date_cols=["linkdt", "linkenddt"],
    )
    out = raw.copy()
    out.columns = [str(c).lower() for c in out.columns]
    out["gvkey"] = out["gvkey"].astype("string")
    out["lpermno"] = pd.to_numeric(out["lpermno"], errors="coerce").astype("Int64")
    out["linkdt"] = pd.to_datetime(out["linkdt"], errors="coerce")
    out["linkenddt"] = pd.to_datetime(out["linkenddt"], errors="coerce")
    out["source_table"] = f"{resolved.library}.{resolved.table}"
    return out, resolved


def cached_compustat_fundamentals(
    db: Any,
    cache_path: Path,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    *,
    gvkeys: Sequence[str] | None = None,
    refresh: bool = False,
) -> tuple[pd.DataFrame, CompustatSource | None, bool]:
    if cache_path.exists() and not refresh:
        return read_parquet_cache(cache_path), None, True
    data, source = pull_compustat_fundamentals(
        db,
        start_date,
        end_date,
        gvkeys=gvkeys,
    )
    write_parquet_cache(data, cache_path)
    return data, source, False


def cached_ccm_links(
    db: Any,
    cache_path: Path,
    *,
    refresh: bool = False,
) -> tuple[pd.DataFrame, CcmSource | None, bool]:
    if cache_path.exists() and not refresh:
        return read_parquet_cache(cache_path), None, True
    data, source = pull_ccm_links(db)
    write_parquet_cache(data, cache_path)
    return data, source, False

"""Secure WRDS discovery and CRSP monthly extraction utilities.

The current CRSP stock format on WRDS is CIZ.  This module discovers the
available library/table at runtime and prefers CIZ ``msf_v2``.  A legacy SIZ
fallback is retained for subscriptions that expose only the older ``msf`` /
``msenames`` interface.

Credentials are intentionally never accepted as function arguments.  The WRDS
Python package uses the user's supported WRDS authentication configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from rdsrp.data.cache import read_parquet_cache, write_parquet_cache


@dataclass(frozen=True)
class CrspSource:
    library: str
    table: str
    format: str
    names_table: str | None = None


PREFERRED_LIBRARIES: tuple[str, ...] = ("crsp_m_stock", "crsp", "crspa")
CIZ_TABLES: tuple[str, ...] = ("msf_v2", "stkmthsecuritydata")
CIZ_INFO_TABLES: tuple[str, ...] = ("stksecurityinfohist",)
LEGACY_NAMES_TABLES: tuple[str, ...] = ("msenames", "stocknames")


def connect_wrds() -> Any:
    try:
        import wrds
    except ImportError as exc:
        raise RuntimeError("The 'wrds' package is required. Install project dependencies first.") from exc
    return wrds.Connection()


def _lower_set(values: Iterable[str]) -> set[str]:
    return {str(value).lower() for value in values}


def available_libraries(db: Any) -> list[str]:
    return sorted(_lower_set(db.list_libraries()))


def resolve_crsp_monthly_source(db: Any) -> CrspSource:
    libraries = available_libraries(db)
    ordered = [lib for lib in PREFERRED_LIBRARIES if lib in libraries]
    ordered.extend(lib for lib in libraries if lib not in ordered and "crsp" in lib and ("stock" in lib or lib in {"crsp", "crspa"}))
    inspected: list[str] = []
    for library in ordered:
        try: tables = _lower_set(db.list_tables(library=library))
        except Exception: continue
        inspected.append(f"{library}({len(tables)} tables)")
        for table in CIZ_TABLES:
            if table in tables:
                if table == "stkmthsecuritydata":
                    info = next((name for name in CIZ_INFO_TABLES if name in tables), None)
                    if info is None: continue
                    return CrspSource(library, table, "ciz", info)
                return CrspSource(library, table, "ciz")
    for library in ordered:
        try: tables = _lower_set(db.list_tables(library=library))
        except Exception: continue
        if "msf" not in tables: continue
        names_table = next((name for name in LEGACY_NAMES_TABLES if name in tables), None)
        if names_table is not None: return CrspSource(library, "msf", "legacy", names_table)
    detail = ", ".join(inspected) if inspected else "no candidate CRSP libraries were accessible"
    raise RuntimeError(f"Could not locate a supported CRSP monthly stock table. Inspected: {detail}. Confirm that your WRDS account has CRSP Stock access.")


def _date_literal(value: str | pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def build_crsp_query(source: CrspSource, start_date: str | pd.Timestamp, end_date: str | pd.Timestamp, ticker: str | None = None, min_price: float | None = None, row_limit: int | None = None, top_n_by_market_cap: int | None = None, include_ticker: str | None = None, universe_formation_date: str | pd.Timestamp | None = None) -> str:
    start = _date_literal(start_date); end = _date_literal(end_date)
    ticker_filter = ""; price_filter = ""; limit_clause = f"\nLIMIT {int(row_limit)}" if row_limit is not None else ""
    if ticker is not None:
        safe_ticker = ticker.upper().replace("'", "''"); ticker_filter = f"\n  AND UPPER(ticker) = '{safe_ticker}'"
    if min_price is not None:
        price_filter = f"\n  AND ABS(mthprc) >= {float(min_price):.8f}" if source.format == "ciz" else f"\n  AND ABS(a.prc) >= {float(min_price):.8f}"
    if source.format == "ciz":
        if source.table == "msf_v2":
            select_columns = """
    permno,
    mthcaldt AS date,
    ticker,
    cusip,
    issuernm AS comnam,
    mthret AS ret,
    mthretx AS retx,
    mthprc AS prc,
    shrout,
    mthcap AS market_equity,
    mthprevcap AS me_lag,
    primaryexch,
    sharetype,
    securitytype,
    securitysubtype,
    usincflg,
    issuertype,
    conditionaltype,
    tradingstatusflg"""
            history_price_filter = "" if top_n_by_market_cap is not None else price_filter
            base = f"""
SELECT{select_columns}
FROM {source.library}.{source.table}
WHERE mthcaldt BETWEEN '{start}' AND '{end}'
  AND sharetype = 'NS'
  AND securitytype = 'EQTY'
  AND securitysubtype = 'COM'
  AND usincflg = 'Y'
  AND issuertype IN ('ACOR', 'CORP')
  AND primaryexch IN ('N', 'A', 'Q')
  AND conditionaltype = 'RW'
  AND tradingstatusflg = 'A'{ticker_filter}{history_price_filter}
""".strip()
            if top_n_by_market_cap is None:
                return f"{base}\nORDER BY mthcaldt, permno{limit_clause}"
            n = int(top_n_by_market_cap)
            if n < 1: raise ValueError("top_n_by_market_cap must be positive.")
            formation = _date_literal(universe_formation_date or start_date)
            safe_include = include_ticker.upper().replace("'", "''") if include_ticker is not None else None
            selected_clause = "permno IN (SELECT permno FROM formation)"
            if safe_include is not None: selected_clause += f" OR UPPER(ticker) = '{safe_include}'"
            formation_price = f"\n  AND ABS(mthprc) >= {float(min_price):.8f}" if min_price is not None else ""
            return f"""
WITH formation AS (
    SELECT permno
    FROM {source.library}.{source.table}
    WHERE mthcaldt = '{formation}'
      AND sharetype = 'NS'
      AND securitytype = 'EQTY'
      AND securitysubtype = 'COM'
      AND usincflg = 'Y'
      AND issuertype IN ('ACOR', 'CORP')
      AND primaryexch IN ('N', 'A', 'Q')
      AND conditionaltype = 'RW'
      AND tradingstatusflg = 'A'{formation_price}
    ORDER BY mthcap DESC NULLS LAST, permno
    LIMIT {n}
), eligible AS (
{base}
)
SELECT *
FROM eligible
WHERE {selected_clause}
ORDER BY date, permno{limit_clause}
""".strip()
        if top_n_by_market_cap is not None: raise ValueError("Per-month top-N extraction is currently supported for CIZ msf_v2.")
        if source.names_table is None: raise ValueError("CIZ security-data source requires stkSecurityInfoHist metadata.")
        ciz_ticker_filter = ticker_filter.replace("ticker", "b.ticker")
        ciz_price_filter = f"\n  AND ABS(a.mthprc) >= {float(min_price):.8f}" if min_price is not None else ""
        return f"""
SELECT
    a.permno,
    a.mthcaldt AS date,
    b.ticker,
    b.cusip,
    b.issuernm AS comnam,
    a.mthret AS ret,
    a.mthretx AS retx,
    a.mthprc AS prc,
    a.shrout,
    a.mthcap AS market_equity,
    a.mthprevcap AS me_lag,
    b.primaryexch,
    b.sharetype,
    b.securitytype,
    b.securitysubtype,
    b.usincflg,
    b.issuertype,
    b.conditionaltype,
    b.tradingstatusflg
FROM {source.library}.{source.table} AS a
JOIN {source.library}.{source.names_table} AS b
  ON a.permno = b.permno
 AND b.secinfostartdt <= a.mthcaldt
 AND a.mthcaldt <= b.secinfoenddt
WHERE a.mthcaldt BETWEEN '{start}' AND '{end}'
  AND b.sharetype = 'NS'
  AND b.securitytype = 'EQTY'
  AND b.securitysubtype = 'COM'
  AND b.usincflg = 'Y'
  AND b.issuertype IN ('ACOR', 'CORP')
  AND b.primaryexch IN ('N', 'A', 'Q')
  AND b.conditionaltype = 'RW'
  AND b.tradingstatusflg = 'A'{ciz_ticker_filter}{ciz_price_filter}
ORDER BY a.mthcaldt, a.permno{limit_clause}
""".strip()
    if source.names_table is None: raise ValueError("Legacy source requires a names/history table.")
    return f"""
SELECT
    a.permno,
    a.date,
    b.ticker,
    b.ncusip AS cusip,
    b.comnam,
    a.ret,
    a.retx,
    ABS(a.prc) AS prc,
    a.shrout,
    ABS(a.prc) * a.shrout AS market_equity,
    NULL::double precision AS me_lag,
    b.exchcd,
    b.shrcd
FROM {source.library}.{source.table} AS a
JOIN {source.library}.{source.names_table} AS b
  ON a.permno = b.permno
 AND b.namedt <= a.date
 AND a.date <= b.nameendt
WHERE a.date BETWEEN '{start}' AND '{end}'
  AND b.shrcd IN (10, 11)
  AND b.exchcd IN (1, 2, 3){ticker_filter.replace('ticker', 'b.ticker')}{price_filter}
ORDER BY a.date, a.permno{limit_clause}
""".strip()


def normalize_crsp_monthly(df: pd.DataFrame, source: CrspSource) -> pd.DataFrame:
    if df.empty: return df.copy()
    out = df.copy(); out.columns = [str(col).lower() for col in out.columns]; out["date"] = pd.to_datetime(out["date"]) + pd.offsets.MonthEnd(0)
    for col in ["permno", "ret", "retx", "prc", "shrout", "market_equity", "me_lag"]:
        if col in out: out[col] = pd.to_numeric(out[col], errors="coerce")
    if "prc" in out: out["prc"] = out["prc"].abs()
    if "market_equity" not in out and {"prc", "shrout"}.issubset(out.columns): out["market_equity"] = out["prc"] * out["shrout"]
    if "me_lag" not in out or out["me_lag"].isna().all():
        out = out.sort_values(["permno", "date"]).reset_index(drop=True); out["me_lag"] = out.groupby("permno", sort=False)["market_equity"].shift(1)
    out["source_format"] = source.format; out["source_table"] = f"{source.library}.{source.table}"
    return out.sort_values(["date", "permno"]).reset_index(drop=True)


def pull_crsp_monthly(db: Any, start_date: str | pd.Timestamp, end_date: str | pd.Timestamp, *, ticker: str | None = None, min_price: float | None = None, row_limit: int | None = None, top_n_by_market_cap: int | None = None, include_ticker: str | None = None, universe_formation_date: str | pd.Timestamp | None = None, source: CrspSource | None = None) -> tuple[pd.DataFrame, CrspSource]:
    resolved = source or resolve_crsp_monthly_source(db)
    query = build_crsp_query(resolved, start_date, end_date, ticker, min_price, row_limit, top_n_by_market_cap, include_ticker, universe_formation_date)
    raw = db.raw_sql(query, date_cols=["date"])
    return normalize_crsp_monthly(raw, resolved), resolved


def cached_crsp_monthly(db: Any, cache_path: Path, start_date: str | pd.Timestamp, end_date: str | pd.Timestamp, *, ticker: str | None = None, min_price: float | None = None, row_limit: int | None = None, top_n_by_market_cap: int | None = None, include_ticker: str | None = None, universe_formation_date: str | pd.Timestamp | None = None, refresh: bool = False) -> tuple[pd.DataFrame, CrspSource | None, bool]:
    if cache_path.exists() and not refresh: return read_parquet_cache(cache_path), None, True
    data, source = pull_crsp_monthly(db, start_date, end_date, ticker=ticker, min_price=min_price, row_limit=row_limit, top_n_by_market_cap=top_n_by_market_cap, include_ticker=include_ticker, universe_formation_date=universe_formation_date)
    write_parquet_cache(data, cache_path)
    return data, source, False

"""Authoritative VIX discovery, extraction, caching, and monthly alignment.

The empirical pipeline prefers the user's WRDS Cboe subscription. WRDS currently
publishes the VIX product under the ``cboe_all`` schema, but the physical table
name is not assumed. If that schema cannot be resolved, the real-data pipeline
can fall back to Cboe's official public VIX historical CSV, with source
provenance retained in the cached data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from rdsrp.data.cache import read_parquet_cache, write_parquet_cache


@dataclass(frozen=True)
class VixSource:
    library: str
    table: str
    date_col: str
    value_col: str


PREFERRED_VIX_LIBRARIES: tuple[str, ...] = ("cboe_all", "cboe")
CBOE_VIX_HISTORY_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"


def _infer_date_col(columns: list[str]) -> str | None:
    lower = {col.lower(): col for col in columns}
    for candidate in ("date", "trade_date", "tradedate", "caldt", "dt"):
        if candidate in lower:
            return lower[candidate]
    return next((col for col in columns if "date" in col.lower()), None)


def _infer_vix_col(columns: list[str]) -> str | None:
    lower = {col.lower(): col for col in columns}
    for candidate in ("vix", "close", "vix_close", "vixclose", "close_price", "closing_value"):
        if candidate in lower:
            return lower[candidate]
    for col in columns:
        name = col.lower()
        if "vix" in name and ("close" in name or name.endswith("vix")):
            return col
    return None


def _source_score(table: str, value_col: str) -> tuple[int, int, str]:
    table_l = table.lower(); value_l = value_col.lower(); score = 0
    if table_l == "vix": score += 10
    elif "vix" in table_l: score += 7
    if value_l == "vix": score += 8
    elif "vix" in value_l: score += 5
    if value_l in {"close", "close_price", "closing_value"} or "close" in value_l: score += 3
    return (-score, len(table_l), table_l)


def resolve_vix_source(db: Any) -> VixSource:
    libraries = {str(x).lower() for x in db.list_libraries()}
    ordered = [lib for lib in PREFERRED_VIX_LIBRARIES if lib in libraries]
    ordered.extend(sorted(lib for lib in libraries if "cboe" in lib and lib not in ordered))
    inspected: list[str] = []; failures: list[str] = []; candidates: list[VixSource] = []; table_inventory: list[str] = []
    for library in ordered:
        try:
            tables = [str(x) for x in db.list_tables(library=library)]
        except Exception as exc:
            failures.append(f"{library}: list_tables failed ({type(exc).__name__})"); continue
        tables = sorted(tables, key=lambda t: ("vix" not in t.lower(), t.lower()))
        table_inventory.extend(f"{library}.{table}" for table in tables)
        for table in tables:
            try:
                sample = db.raw_sql(f"SELECT * FROM {library}.{table} LIMIT 5")
            except Exception as exc:
                failures.append(f"{library}.{table}: {type(exc).__name__}"); continue
            columns = [str(c) for c in sample.columns]; date_col = _infer_date_col(columns); value_col = _infer_vix_col(columns)
            inspected.append(f"{library}.{table}({', '.join(columns)})")
            if date_col and value_col: candidates.append(VixSource(library, table, date_col, value_col))
    if candidates:
        return sorted(candidates, key=lambda src: _source_score(src.table, src.value_col))[0]
    detail_parts = []
    if table_inventory: detail_parts.append("tables=" + ", ".join(table_inventory[:30]))
    if inspected: detail_parts.append("inspected=" + "; ".join(inspected[:15]))
    if failures: detail_parts.append("failures=" + "; ".join(failures[:15]))
    detail = " | ".join(detail_parts) if detail_parts else "no readable Cboe tables were exposed"
    raise RuntimeError("Could not resolve the Cboe VIX table through WRDS. WRDS currently lists Daily pricing of the VIX under schema 'cboe_all'. " f"Discovery detail: {detail}.")


def pull_vix_daily(db: Any, start_date: str | pd.Timestamp, end_date: str | pd.Timestamp, *, source: VixSource | None = None) -> tuple[pd.DataFrame, VixSource]:
    resolved = source or resolve_vix_source(db)
    start = pd.Timestamp(start_date).strftime("%Y-%m-%d"); end = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    query = f"""SELECT {resolved.date_col} AS date, {resolved.value_col} AS vix
FROM {resolved.library}.{resolved.table}
WHERE {resolved.date_col} BETWEEN '{start}' AND '{end}'
ORDER BY {resolved.date_col}""".strip()
    data = db.raw_sql(query, date_cols=["date"]); data.columns = [str(c).lower() for c in data.columns]
    data["date"] = pd.to_datetime(data["date"]); data["vix"] = pd.to_numeric(data["vix"], errors="coerce")
    data = data.dropna(subset=["date", "vix"]).sort_values("date").reset_index(drop=True)
    data["source_table"] = f"{resolved.library}.{resolved.table}"; data["source_value_col"] = resolved.value_col; data["source_type"] = "wrds_cboe"
    return data, resolved


def load_cboe_vix_history(start_date: str | pd.Timestamp, end_date: str | pd.Timestamp, *, url: str = CBOE_VIX_HISTORY_URL) -> pd.DataFrame:
    try:
        raw = pd.read_csv(url)
    except Exception as exc:
        raise RuntimeError("WRDS VIX discovery failed and the official Cboe VIX CSV could not be downloaded. " f"Download the file manually from {CBOE_VIX_HISTORY_URL} and use the local VIX loader. " f"Underlying download error: {type(exc).__name__}: {exc}") from exc
    columns = {str(c).strip().lower(): str(c) for c in raw.columns}; date_col = columns.get("date"); value_col = columns.get("close") or columns.get("vix")
    if date_col is None or value_col is None:
        raise RuntimeError(f"The official Cboe VIX CSV schema was not recognized. Observed columns: {list(raw.columns)}")
    out = raw[[date_col, value_col]].rename(columns={date_col: "date", value_col: "vix"}).copy(); out["date"] = pd.to_datetime(out["date"], errors="coerce"); out["vix"] = pd.to_numeric(out["vix"], errors="coerce")
    start = pd.Timestamp(start_date); end = pd.Timestamp(end_date); out = out.dropna(subset=["date", "vix"]); out = out[(out["date"] >= start) & (out["date"] <= end)].sort_values("date").reset_index(drop=True)
    if out.empty: raise RuntimeError(f"The official Cboe VIX CSV contained no observations between {start.date()} and {end.date()}.")
    out["source_table"] = "Cboe official VIX historical data"; out["source_value_col"] = value_col; out["source_type"] = "cboe_public_csv"; out["source_url"] = url
    return out


def cached_vix_daily(db: Any, cache_path: Path, start_date: str | pd.Timestamp, end_date: str | pd.Timestamp, *, refresh: bool = False, allow_public_fallback: bool = True) -> tuple[pd.DataFrame, VixSource | None, bool]:
    if cache_path.exists() and not refresh: return read_parquet_cache(cache_path), None, True
    try:
        data, source = pull_vix_daily(db, start_date, end_date)
    except RuntimeError as wrds_exc:
        if not allow_public_fallback: raise
        print("WRDS Cboe VIX table could not be resolved; using Cboe's official VIX historical CSV as the authoritative fallback.")
        print(f"WRDS VIX discovery detail: {wrds_exc}")
        data = load_cboe_vix_history(start_date, end_date); source = None
    write_parquet_cache(data, cache_path); return data, source, False


def vix_daily_to_monthly(vix_daily: pd.DataFrame) -> pd.DataFrame:
    if vix_daily.empty: return pd.DataFrame(columns=["date", "vix"])
    if not {"date", "vix"}.issubset(vix_daily.columns): raise ValueError("Daily VIX data must contain 'date' and 'vix'.")
    frame = vix_daily[["date", "vix"]].copy(); frame["date"] = pd.to_datetime(frame["date"]); frame["vix"] = pd.to_numeric(frame["vix"], errors="coerce"); frame = frame.dropna().sort_values("date"); frame["month"] = frame["date"].dt.to_period("M")
    monthly = frame.groupby("month", as_index=False).tail(1).copy(); monthly["date"] = monthly["month"].dt.to_timestamp("M")
    return monthly[["date", "vix"]].sort_values("date").reset_index(drop=True)


def load_vix_csv(path: Path, date_col: str = "date", vix_col: str = "vix") -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col not in df or vix_col not in df: raise ValueError(f"VIX file must contain '{date_col}' and '{vix_col}'.")
    out = df[[date_col, vix_col]].rename(columns={date_col: "date", vix_col: "vix"}); out["date"] = pd.to_datetime(out["date"]); out["vix"] = pd.to_numeric(out["vix"], errors="coerce")
    return out.dropna().sort_values("date").reset_index(drop=True)

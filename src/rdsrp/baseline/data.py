"""Canonical baseline-panel loading and validation."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from rdsrp.model_spec import BASELINE_FEATURES, BASELINE_TARGET, FORMATION_DATE, REALIZED_RETURN_DATE

BASELINE_PREDICTORS: tuple[str, ...] = BASELINE_FEATURES
CANONICAL_PANEL_RELATIVE_PATH = Path("data") / "processed" / "baseline_modeling_panel.parquet"
EXPECTED_PRODUCTION_COUNTS = {
    "total_stock_months": 2_140_091,
    "complete_predictor_rows": 1_798_891,
    "final_screened_rows": 1_340_823,
    "duplicate_permno_date_rows": 0,
}
BENCHMARK_NAME = "pooled_expanding_historical_mean"
BENCHMARK_DEFINITION = (
    "At formation month t, every stock receives the pooled mean of all next-month returns "
    "whose targets are observable by t."
)


def canonical_panel_path(repo_root: Path) -> Path:
    return repo_root / CANONICAL_PANEL_RELATIVE_PATH


def load_run_config(repo_root: Path, mode: str) -> dict[str, Any]:
    if mode not in {"quick", "production"}:
        raise ValueError("mode must be quick or production")
    path = repo_root / "configs" / f"{mode}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")
    return data


def load_production_config(repo_root: Path) -> dict[str, Any]:
    return load_run_config(repo_root, "production")


def _month_end(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values) + pd.offsets.MonthEnd(0)


def _formation(panel: pd.DataFrame) -> pd.Series:
    if FORMATION_DATE not in panel:
        raise ValueError("Canonical panel must contain formation_date")
    formation = _month_end(panel[FORMATION_DATE])
    if "date" in panel:
        date = _month_end(panel["date"])
        if (formation.notna() & date.notna() & formation.ne(date)).any():
            raise AssertionError("date and formation_date disagree")
    return formation


def _eligible(panel: pd.DataFrame, min_price: float) -> pd.Series:
    if "price_eligible" in panel:
        return panel["price_eligible"].fillna(False).astype(bool)
    if "prc" in panel:
        return pd.to_numeric(panel["prc"], errors="coerce").abs().ge(min_price)
    raise ValueError("Canonical panel must contain price_eligible or prc")


def production_count_audit(
    panel: pd.DataFrame,
    *,
    min_price: float = 5.0,
    expected: Mapping[str, int] | None = EXPECTED_PRODUCTION_COUNTS,
    strict: bool = True,
) -> dict[str, int]:
    formation = _formation(panel)
    required = {"permno", REALIZED_RETURN_DATE, BASELINE_TARGET, "regime", "vix", *BASELINE_PREDICTORS}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Completed production panel is missing required columns: {sorted(missing)}")
    complete = panel[list(BASELINE_PREDICTORS)].notna().all(axis=1)
    final = (
        complete & _eligible(panel, min_price) & panel[BASELINE_TARGET].notna()
        & panel["regime"].notna() & panel["vix"].notna()
        & formation.notna() & panel[REALIZED_RETURN_DATE].notna()
    )
    keys = pd.DataFrame({"permno": panel["permno"], FORMATION_DATE: formation})
    audit = {
        "total_stock_months": int(len(panel)),
        "complete_predictor_rows": int(complete.sum()),
        "final_screened_rows": int(final.sum()),
        "duplicate_permno_date_rows": int(keys.duplicated(["permno", FORMATION_DATE]).sum()),
    }
    if strict and expected is not None:
        bad = {k: (audit[k], int(v)) for k, v in expected.items() if audit[k] != int(v)}
        if bad:
            details = ", ".join(f"{k}: observed={a:,} expected={b:,}" for k, (a, b) in bad.items())
            raise RuntimeError(f"Completed production-panel counts differ from validated snapshot; {details}")
    return audit


def prepare_baseline_sample(panel: pd.DataFrame, *, min_price: float = 5.0) -> pd.DataFrame:
    formation = _formation(panel)
    required = {"permno", REALIZED_RETURN_DATE, BASELINE_TARGET, "regime", "vix", *BASELINE_PREDICTORS}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Canonical panel is missing required model fields: {sorted(missing)}")
    keep = (
        _eligible(panel, min_price)
        & panel[list(BASELINE_PREDICTORS)].notna().all(axis=1)
        & panel[BASELINE_TARGET].notna()
        & panel["regime"].notna()
        & panel["vix"].notna()
        & formation.notna()
        & panel[REALIZED_RETURN_DATE].notna()
    )
    if "baseline_complete_case" in panel:
        flag = panel["baseline_complete_case"].fillna(False).astype(bool)
        if flag.ne(keep).any():
            raise AssertionError("baseline_complete_case disagrees with frozen model eligibility")
        keep = flag
    wanted = [
        "date", FORMATION_DATE, REALIZED_RETURN_DATE, "permno", "ticker", "regime", "vix",
        *BASELINE_PREDICTORS, BASELINE_TARGET, "prc", "price_eligible", "baseline_complete_case",
        "me_lag", "me_lag_date", "market_equity", "vix_expanding_median",
    ]
    sample = panel.loc[keep, [c for c in dict.fromkeys(wanted) if c in panel]].copy()
    sample[FORMATION_DATE] = formation.loc[keep].to_numpy()
    sample["date"] = sample[FORMATION_DATE]
    sample[REALIZED_RETURN_DATE] = _month_end(sample[REALIZED_RETURN_DATE])
    expected_realized = sample[FORMATION_DATE] + pd.offsets.MonthEnd(1)
    if not sample[REALIZED_RETURN_DATE].eq(expected_realized).all():
        raise AssertionError("next_month_return must be realized in calendar month t+1")
    if sample.duplicated(["permno", FORMATION_DATE]).any():
        raise AssertionError("Baseline sample contains duplicate permno/formation_date rows")
    regimes = set(sample["regime"].astype(str).str.upper().unique())
    if not regimes.issubset({"HIGH", "LOW"}):
        raise ValueError(f"Unexpected volatility-regime labels: {sorted(regimes)}")
    for col in (*BASELINE_PREDICTORS, BASELINE_TARGET, "vix"):
        sample[col] = pd.to_numeric(sample[col], errors="raise").astype("float32")
    if not np.isfinite(sample[[*BASELINE_PREDICTORS, BASELINE_TARGET, "vix"]].to_numpy(float)).all():
        raise ValueError("Canonical baseline sample contains non-finite model inputs")
    return sample.sort_values([FORMATION_DATE, "permno"], kind="mergesort").reset_index(drop=True)


def read_production_panel(panel_path: Path) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read the canonical baseline Parquet panel") from exc
    schema = set(pq.ParquetFile(panel_path).schema.names)
    mandatory = {
        "permno", FORMATION_DATE, REALIZED_RETURN_DATE, BASELINE_TARGET, "regime", "vix",
        "price_eligible", "baseline_complete_case", *BASELINE_PREDICTORS,
    }
    missing = mandatory.difference(schema)
    if missing:
        raise ValueError(f"Canonical Parquet is missing columns: {sorted(missing)}")
    optional = ["date", "ticker", "prc", "me_lag", "me_lag_date", "market_equity", "vix_expanding_median"]
    columns = list(mandatory) + [c for c in optional if c in schema]
    return pd.read_parquet(panel_path, columns=list(dict.fromkeys(columns)))

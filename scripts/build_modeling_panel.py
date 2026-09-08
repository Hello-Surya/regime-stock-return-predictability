"""Build the leakage-safe CRSP-Compustat-CCM baseline predictor panel."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdsrp.data.accounting import (  # noqa: E402
    cached_ccm_links,
    cached_compustat_fundamentals,
)
from rdsrp.data.modeling_panel import (  # noqa: E402
    PREDICTORS,
    build_complete_predictor_panel,
    construction_summary,
    distribution_report,
    sample_flow,
    validate_economic_sanity,
)
from rdsrp.features.book_equity import (  # noqa: E402
    VALID_CCM_LINK_PRIMARY,
    VALID_CCM_LINK_TYPES,
    filter_ccm_links,
)
from rdsrp.data.vix import cached_vix_daily, vix_daily_to_monthly  # noqa: E402
from rdsrp.data.wrds import cached_crsp_monthly, connect_wrds  # noqa: E402
from rdsrp.regimes.vix_regime import label_vix_regime  # noqa: E402


def _load_config(mode: str) -> dict[str, Any]:
    path = REPO_ROOT / "configs" / f"{mode}.yaml"
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid configuration: {path}")
    return config


def _subtract_years(date_value: str, years: int) -> str:
    ts = pd.Timestamp(date_value)
    try:
        return str(ts.replace(year=ts.year - years).date())
    except ValueError:
        return str(ts.replace(month=2, day=28, year=ts.year - years).date())


def _cache_source_name(frame: pd.DataFrame, fallback: str = "local cache") -> str:
    if "source_table" in frame.columns and frame["source_table"].notna().any():
        return str(frame.loc[frame["source_table"].notna(), "source_table"].iloc[0])
    return fallback


def _git_state() -> tuple[str | None, bool | None]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return sha, dirty
    except Exception:
        return None, None




def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without requiring pandas' optional tabulate dependency."""
    if frame.empty:
        return "No valid distribution rows."
    columns = [str(c) for c in frame.columns]

    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            text = f"{value:.10g}"
        else:
            text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def _write_summary_markdown(
    path: Path,
    summary: dict[str, Any],
    sanity: dict[str, Any],
    distributions: pd.DataFrame,
) -> None:
    lines = [
        "# Data Construction Validation Summary",
        "",
        "This report describes the CRSP-Compustat-CCM baseline predictor panel. "
        "It contains aggregate diagnostics only; licensed row-level WRDS data are not included.",
        "",
        "## Panel coverage",
        "",
        f"- Stock-month observations: {summary['panel_rows']:,}",
        f"- Unique PERMNOs: {summary['panel_unique_permnos']:,}",
        f"- Sample: {summary['panel_start']} through {summary['panel_end']}",
        f"- Valid book-to-market months: {summary['valid_book_to_market_months']:,}",
        f"- Complete baseline predictor rows: {summary['complete_predictor_rows']:,}",
        f"- Duplicate PERMNO/date rows: {summary['duplicate_permno_date_rows']:,}",
        "",
        "## Coverage fractions",
        "",
        f"- log_me: {summary['coverage_log_me']:.4f}",
        f"- mom_12_2: {summary['coverage_mom_12_2']:.4f}",
        f"- book_to_market: {summary['coverage_book_to_market']:.4f}",
        f"- next_month_return: {summary['coverage_next_month_return']:.4f}",
        f"- regime: {summary['coverage_regime']:.4f}",
        "",
        "## Link and unit checks",
        "",
        f"- CCM rows retained after baseline link filters: {summary.get('ccm_rows_retained', 0):,}",
        f"- Accounting firm-years unmatched to a valid CCM link: {summary.get('accounting_rows_unmatched_ccm', 0):,}",
        f"- GVKEY-years with multiple eligible PERMNOs: {summary.get('gvkey_years_multiple_permnos', 0):,}",
        f"- GVKEY-years with incomplete December ME: {summary.get('gvkey_years_incomplete_december_me', 0):,}",
        f"- B/M unit sanity: {sanity.get('unit_sanity_status')}",
        f"- B/M median: {sanity.get('book_to_market_median')}",
        f"- B/M 99.9th percentile: {sanity.get('book_to_market_p999')}",
        "",
        "## Distribution percentiles",
        "",
        _markdown_table(distributions),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build(mode: str, *, refresh: bool = False) -> Path:
    config = _load_config(mode)
    real_cfg = config["run"]["real"]
    construction_cfg = config["run"].get("data_construction", {})
    configured_link_types = tuple(construction_cfg.get("ccm_link_types", VALID_CCM_LINK_TYPES))
    configured_link_primary = tuple(construction_cfg.get("ccm_link_primary", VALID_CCM_LINK_PRIMARY))
    if configured_link_types != VALID_CCM_LINK_TYPES or configured_link_primary != VALID_CCM_LINK_PRIMARY:
        raise ValueError("Run configuration does not match the frozen CCM-link methodology.")
    if int(construction_cfg.get("accounting_assignment_month", 6)) != 6:
        raise ValueError("Run configuration does not match the frozen June accounting assignment convention.")
    sample_start = str(real_cfg["crsp_start"])
    sample_end = str(real_cfg["crsp_end"])
    crsp_history_years = int(construction_cfg.get("crsp_history_years", 2))
    accounting_history_years = int(construction_cfg.get("accounting_history_years", 3))
    crsp_extract_start = _subtract_years(sample_start, crsp_history_years)
    compustat_start = _subtract_years(sample_start, accounting_history_years)
    min_price = float(real_cfg["min_price"])

    validation_dir = REPO_ROOT / "results" / "data_validation" / mode
    validation_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = REPO_ROOT / "data" / "raw"
    processed_dir = REPO_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    top_n = real_cfg.get("top_n_by_market_cap")
    universe_tag = f"fixedtop{int(top_n)}" if top_n is not None else "full"
    crsp_cache = raw_dir / "crsp" / f"monthly_{crsp_extract_start[:4]}_{sample_end[:4]}_{universe_tag}.parquet"
    ccm_cache = raw_dir / "ccm" / "ccm_link_history.parquet"
    comp_cache = raw_dir / "compustat" / f"funda_{compustat_start[:4]}_{sample_end[:4]}_{universe_tag}.parquet"
    vix_cache = raw_dir / "vix" / f"daily_{real_cfg['vix_history_start'][:4]}_{sample_end[:4]}.parquet"

    print("[1/9] Loading research configuration...")
    print(f"Mode: {mode}; research sample: {sample_start} to {sample_end}")
    print(f"Construction history: CRSP from {crsp_extract_start}; Compustat from {compustat_start}")

    db = None
    crsp_source_name = "local cache"
    comp_source_name = "local cache"
    ccm_source_name = "local cache"
    vix_source_name = "local cache"
    try:
        db = connect_wrds()

        print("[2/9] Loading or extracting CRSP data...")
        crsp, crsp_source, _ = cached_crsp_monthly(
            db,
            crsp_cache,
            crsp_extract_start,
            sample_end,
            min_price=None,
            top_n_by_market_cap=int(top_n) if top_n is not None else None,
            universe_formation_date=real_cfg.get("universe_formation_date"),
            refresh=refresh,
        )
        if crsp_source is not None:
            crsp_source_name = f"{crsp_source.library}.{crsp_source.table} ({crsp_source.format.upper()})"
        else:
            crsp_source_name = _cache_source_name(crsp)
        if crsp.empty:
            raise RuntimeError("CRSP extraction returned zero rows.")

        print("[3/9] Loading or extracting CCM links...")
        ccm, ccm_source, _ = cached_ccm_links(db, ccm_cache, refresh=refresh)
        if ccm_source is not None:
            ccm_source_name = f"{ccm_source.library}.{ccm_source.table}"
        else:
            ccm_source_name = _cache_source_name(ccm)
        if ccm.empty:
            raise RuntimeError("CCM extraction returned zero rows.")

        print("[4/9] Loading or extracting Compustat fundamentals...")
        valid_ccm, _ = filter_ccm_links(ccm)
        crsp_permnos = set(pd.to_numeric(crsp["permno"], errors="coerce").dropna().astype(int))
        candidate_gvkeys = sorted(
            valid_ccm.loc[
                pd.to_numeric(valid_ccm["lpermno"], errors="coerce").isin(crsp_permnos), "gvkey"
            ]
            .dropna()
            .astype(str)
            .unique()
        )
        if top_n is not None and not candidate_gvkeys:
            raise RuntimeError("No candidate Compustat GVKEYs were found for the quick CRSP universe.")
        query_gvkeys = candidate_gvkeys if top_n is not None else None
        comp, comp_source, _ = cached_compustat_fundamentals(
            db,
            comp_cache,
            compustat_start,
            sample_end,
            gvkeys=query_gvkeys,
            refresh=refresh,
        )
        if comp_source is not None:
            comp_source_name = f"{comp_source.library}.{comp_source.table}"
        else:
            comp_source_name = _cache_source_name(comp)
        if comp.empty:
            raise RuntimeError("Compustat extraction returned zero rows.")

        print("[5/9] Loading VIX history and constructing monthly regimes...")
        vix_daily, vix_source, _ = cached_vix_daily(
            db,
            vix_cache,
            real_cfg["vix_history_start"],
            sample_end,
            refresh=refresh,
        )
        if vix_source is not None:
            vix_source_name = f"{vix_source.library}.{vix_source.table}"
        else:
            vix_source_name = _cache_source_name(vix_daily)
    finally:
        if db is not None:
            db.close()

    print("[6/9] Constructing book equity and applying CCM/accounting timing rules...")
    vix_monthly = vix_daily_to_monthly(vix_daily)
    regimes = label_vix_regime(vix_monthly)
    panel_all, audit = build_complete_predictor_panel(
        crsp,
        comp,
        ccm,
        regimes,
        min_price=min_price,
    )

    print("[7/9] Constructing book-to-market and complete predictor panel...")
    panel = panel_all[
        (panel_all["date"] >= pd.Timestamp(sample_start))
        & (panel_all["date"] <= pd.Timestamp(sample_end))
    ].copy()
    if panel.empty:
        raise RuntimeError("Final research-date panel is empty.")
    if panel.duplicated(["permno", "date"]).any():
        raise RuntimeError("Final panel contains duplicate PERMNO-date observations.")
    required = {"permno", "date", "log_me", "book_to_market", "mom_12_2", "next_month_return"}
    missing = required.difference(panel.columns)
    if missing:
        raise RuntimeError(f"Final panel is missing required fields: {sorted(missing)}")

    print("[8/9] Running data validation and writing aggregate artifacts...")
    distributions = distribution_report(panel)
    flow = sample_flow(panel, min_price=min_price)
    sanity = validate_economic_sanity(panel)
    summary = construction_summary(comp, ccm, panel, audit)
    summary.update({f"sanity_{k}": v for k, v in sanity.items()})

    panel_path = processed_dir / ("modeling_panel.parquet" if mode == "production" else "modeling_panel_quick.parquet")
    panel.to_parquet(panel_path, index=False)
    distributions.to_csv(validation_dir / "distribution_percentiles.csv", index=False)
    flow.to_csv(validation_dir / "sample_flow.csv", index=False)
    with (validation_dir / "accounting_data_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=str)
    _write_summary_markdown(validation_dir / "data_construction_summary.md", summary, sanity, distributions)

    git_sha, git_dirty = _git_state()
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "research_sample_start": sample_start,
        "research_sample_end": sample_end,
        "crsp_extraction_start": crsp_extract_start,
        "compustat_extraction_start": compustat_start,
        "crsp_source": crsp_source_name,
        "compustat_source": comp_source_name,
        "ccm_source": ccm_source_name,
        "vix_source": vix_source_name,
        "book_equity_construction": "SHE + deferred taxes/ITC - preferred stock; SHE=SEQ, else CEQ+PSTK, else AT-LT",
        "preferred_stock_hierarchy": "PSTKRV, then PSTKL, then PSTK; explicit zero if all missing",
        "deferred_tax_hierarchy": "TXDITC, then TXDB+ITCB; explicit zero if all missing",
        "ccm_link_types": ["LC", "LU"],
        "ccm_link_primary": ["P", "C"],
        "accounting_timing": "fiscal-year-end calendar y-1 first eligible June y; active June y through May y+1",
        "book_to_market_denominator": "firm-level sum of prior-December CRSP ME across eligible linked PERMNOs",
        "crsp_market_equity_units": "USD thousands",
        "book_equity_and_me_dec_units": "USD millions",
        "predictors": list(PREDICTORS),
        "git_commit_at_build": git_sha,
        "git_working_tree_dirty": git_dirty,
        "counts": {
            "raw_crsp_rows_with_history": int(len(crsp)),
            "raw_compustat_rows": int(len(comp)),
            "raw_ccm_rows": int(len(ccm)),
            "final_panel_rows": int(len(panel)),
            "unique_permnos": int(panel["permno"].nunique()),
            "valid_book_to_market_rows": int(panel["book_to_market"].notna().sum()),
            "complete_predictor_rows": int(panel[list(PREDICTORS)].notna().all(axis=1).sum()),
        },
    }
    with (validation_dir / "build_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, default=str)

    print("[9/9] Data construction completed successfully.")
    complete = panel[list(PREDICTORS)].notna().all(axis=1)
    print(f"Panel rows: {len(panel):,}")
    print(f"Unique securities: {panel['permno'].nunique():,}")
    print(f"Sample: {panel['date'].min().date()} to {panel['date'].max().date()}")
    print(f"Valid size: {panel['log_me'].notna().sum():,}")
    print(f"Valid momentum: {panel['mom_12_2'].notna().sum():,}")
    print(f"Valid book-to-market: {panel['book_to_market'].notna().sum():,}")
    print(f"All three predictors: {complete.sum():,}")
    print(f"Duplicate permno/date rows: {panel.duplicated(['permno', 'date']).sum():,}")
    print(f"Processed panel: {panel_path}")
    print(f"Validation artifacts: {validation_dir}")
    return panel_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Build the reduced research universe.")
    mode.add_argument("--production", action="store_true", help="Build the full research universe.")
    parser.add_argument("--refresh", action="store_true", help="Refresh local WRDS caches.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mode = "production" if args.production else "quick"
    build(mode, refresh=args.refresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

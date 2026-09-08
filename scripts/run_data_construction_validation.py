"""Run the complete local validation sequence for baseline data construction."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"


def _run(args: list[str], label: str) -> None:
    print(f"\n=== {label} ===", flush=True)
    subprocess.run([str(PYTHON), *args], cwd=REPO_ROOT, check=True)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    if not PYTHON.exists():
        raise FileNotFoundError(f"Expected project interpreter not found: {PYTHON}")

    print("DATA CONSTRUCTION VALIDATION: BEGIN", flush=True)
    _run(["scripts/validate_accounting_sources.py"], "LIVE WRDS ACCOUNTING SOURCE CHECK")
    _run(
        [
            "-m",
            "pytest",
            "-q",
            "tests/test_book_equity.py",
            "tests/test_modeling_panel.py",
            "tests/test_accounting_extraction.py",
            "tests/test_feature_timing.py",
            "tests/test_cross_sectional_portfolios.py",
        ],
        "FOCUSED TESTS",
    )
    _run(["-m", "pytest", "-q"], "COMPLETE TEST SUITE")
    _run(["scripts/build_modeling_panel.py", "--quick"], "REAL-DATA QUICK BUILD")
    _run(["scripts/build_modeling_panel.py", "--production"], "REAL-DATA PRODUCTION BUILD")

    validation_dir = REPO_ROOT / "results" / "data_validation" / "production"
    summary = _load_json(validation_dir / "accounting_data_validation.json")
    metadata = _load_json(validation_dir / "build_metadata.json")

    print("\n=== COPY FROM HERE ===")
    print("DATA_CONSTRUCTION_VALIDATION=PASS")
    print(f"CRSP_SOURCE={metadata.get('crsp_source')}")
    print(f"COMPUSTAT_SOURCE={metadata.get('compustat_source')}")
    print(f"CCM_SOURCE={metadata.get('ccm_source')}")
    print(f"VIX_SOURCE={metadata.get('vix_source')}")
    print(f"SAMPLE={summary.get('panel_start')}..{summary.get('panel_end')}")
    print(f"PANEL_ROWS={summary.get('panel_rows')}")
    print(f"UNIQUE_PERMNOS={summary.get('panel_unique_permnos')}")
    print(f"VALID_BM_MONTHS={summary.get('valid_book_to_market_months')}")
    print(f"COMPLETE_PREDICTOR_ROWS={summary.get('complete_predictor_rows')}")
    print(f"DUPLICATE_PERMNO_DATE={summary.get('duplicate_permno_date_rows')}")
    print(f"CCM_RETAINED={summary.get('ccm_rows_retained')}")
    print(f"ACCOUNTING_UNMATCHED_CCM={summary.get('accounting_rows_unmatched_ccm')}")
    print(f"BM_COVERAGE={summary.get('coverage_book_to_market')}")
    print(f"LOG_ME_COVERAGE={summary.get('coverage_log_me')}")
    print(f"MOMENTUM_COVERAGE={summary.get('coverage_mom_12_2')}")
    print(f"BM_MEDIAN={summary.get('sanity_book_to_market_median')}")
    print(f"BM_P999={summary.get('sanity_book_to_market_p999')}")
    print(f"UNIT_SANITY={summary.get('sanity_unit_sanity_status')}")
    print("=== COPY TO HERE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

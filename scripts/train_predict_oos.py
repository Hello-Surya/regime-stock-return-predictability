"""Canonical baseline OOS training entry point."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdsrp.baseline import (  # noqa: E402
    run_production_baseline,
    run_quick_baseline,
    run_software_validation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--quick",
        action="store_true",
        help="Run reduced real-data estimation using data/processed/baseline_modeling_panel.parquet.",
    )
    mode.add_argument(
        "--production",
        action="store_true",
        help="Run the full frozen baseline OOS estimation on the canonical panel.",
    )
    mode.add_argument(
        "--validate",
        action="store_true",
        help="Run synthetic software validation only; no empirical interpretation.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Clear/recompute model-selection and monthly prediction checkpoints.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.validate:
        run_software_validation(REPO_ROOT)
    elif args.quick:
        run_quick_baseline(REPO_ROOT, refresh=args.refresh)
    else:
        run_production_baseline(REPO_ROOT, refresh=args.refresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

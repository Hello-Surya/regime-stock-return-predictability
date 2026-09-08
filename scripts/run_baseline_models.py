"""Run production baseline Elastic Net and XGBoost estimation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdsrp.baseline import run_production_baseline, run_software_validation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--production", action="store_true", help="Run full production OOS estimation.")
    mode.add_argument(
        "--validate",
        action="store_true",
        help="Run synthetic computational/software validation only; not empirical results.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Explicitly clear/recompute model-selection and monthly prediction checkpoints.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.validate:
        run_software_validation()
    else:
        run_production_baseline(REPO_ROOT, refresh=args.refresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

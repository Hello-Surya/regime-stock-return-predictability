"""Run preliminary cross-sectional portfolio evaluation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdsrp.logging_config import setup_logging  # noqa: E402
from rdsrp.portfolios.strict_evaluation import run_strict_cross_sectional_evaluation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use the computationally reduced preliminary cross-sectional configuration.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Intentionally refresh cached WRDS CRSP/VIX data before evaluation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    if not args.quick:
        print("The current cross-sectional research stage is implemented for the quick-run configuration.")
        print("Use --quick. The production run follows complete Compustat/CCM data construction.")
        return 2
    try:
        artifacts = run_strict_cross_sectional_evaluation(
            REPO_ROOT, quick=True, refresh=args.refresh
        )
    except Exception as exc:
        print(f"Cross-sectional research run failed: {exc}")
        return 1
    print(f"Cross-sectional portfolio evaluation complete: {artifacts['output_dir']}")
    print(f"Portfolio summary: {artifacts['summary']}")
    print(f"Rank summary: {artifacts['rank_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

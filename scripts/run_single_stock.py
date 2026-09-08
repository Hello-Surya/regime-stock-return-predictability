"""Run single-stock OOS validation from a cross-sectional training sample."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdsrp.logging_config import setup_logging  # noqa: E402
from rdsrp.pipeline import run_synthetic_validation, run_wrds_single_stock  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AAPL", help="Ticker to display in final OOS output.")
    parser.add_argument("--synthetic", action="store_true", help="Run generated-data software validation only.")
    parser.add_argument("--quick", action="store_true", help="Use the computationally reduced preliminary empirical configuration.")
    parser.add_argument("--refresh", action="store_true", help="Refresh cached WRDS CRSP/VIX files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    if args.synthetic:
        ticker = args.ticker if args.ticker != "AAPL" else "STK001"
        artifacts = run_synthetic_validation(REPO_ROOT, ticker=ticker)
        print(f"Synthetic validation complete: {artifacts['output_dir']}")
        print("These outputs are synthetic software validation only, not empirical results.")
        return 0

    if not args.quick:
        print("The complete production run requires the Compustat/CCM book-to-market milestones.")
        print("For the current real preliminary baseline use --quick.")
        return 2

    try:
        artifacts = run_wrds_single_stock(
            REPO_ROOT,
            ticker=args.ticker,
            quick=True,
            refresh=args.refresh,
        )
    except Exception as exc:
        print(f"Research run failed: {exc}")
        return 1
    print(f"Preliminary empirical run complete: {artifacts['output_dir']}")
    print(f"Predictions: {artifacts['predictions']}")
    print(f"Metrics: {artifacts['metrics']}")
    print(f"Summary: {artifacts['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

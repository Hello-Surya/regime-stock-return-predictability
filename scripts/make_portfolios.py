"""Run cross-sectional portfolio evaluation from validated research artifacts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdsrp.baseline.economic_value_three_regime import (  # noqa: E402
    run_three_regime_economic_value,
)
from rdsrp.logging_config import setup_logging  # noqa: E402
from rdsrp.portfolios.strict_evaluation import run_strict_cross_sectional_evaluation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--production",
        action="store_true",
        help="Evaluate frozen production OOS predictions across binary and three-state VIX regimes.",
    )
    mode.add_argument(
        "--quick",
        action="store_true",
        help="Run the historical reduced two-predictor cross-sectional validation path.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh cached WRDS data only for the historical --quick path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    try:
        if args.production:
            if args.refresh:
                raise ValueError("--refresh is not used with --production; production reuses saved artifacts.")
            artifacts = run_three_regime_economic_value(REPO_ROOT)
            print()
            print(f"Economic-value analysis complete: {artifacts['output_dir']}")
            print(f"Three-regime summary: {artifacts['three_regime_summary']}")
            print(f"Regime validation: {artifacts['regime_validation']}")
            return 0

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

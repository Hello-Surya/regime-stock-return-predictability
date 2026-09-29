"""Run formal HAC inference on frozen monthly economic-value outputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdsrp.inference.pipeline import run_formal_inference  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--production",
        action="store_true",
        required=True,
        help="Run formal inference on the frozen production monthly outputs.",
    )
    return parser.parse_args()


def main() -> int:
    parse_args()
    try:
        artifacts = run_formal_inference(REPO_ROOT)
    except Exception as exc:
        print(f"Formal inference run failed: {exc}")
        return 1
    print()
    print(f"Formal inference complete: {artifacts['output_dir']}")
    print(f"D10-D1 inference: {artifacts['long_short_inference']}")
    print(f"Regime differences: {artifacts['long_short_differences']}")
    print(f"Rank-IC inference: {artifacts['rank_ic_inference']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

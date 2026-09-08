"""Top-level pipeline entry point."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--ticker", default="STK001")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.synthetic:
        return subprocess.call([sys.executable, str(root / "scripts" / "run_single_stock.py"), "--synthetic", "--ticker", args.ticker])
    print("Real-data orchestration is available through the current empirical pipeline.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

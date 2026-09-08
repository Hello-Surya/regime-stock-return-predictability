"""Build monthly predictors from a cached CRSP-shaped parquet file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rdsrp.features.build import build_features  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import pandas as pd
    df = pd.read_parquet(args.input)
    out = build_features(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"Wrote {len(out):,} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

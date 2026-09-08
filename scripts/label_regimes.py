"""Label monthly VIX observations using the expanding historical median."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rdsrp.regimes.vix_regime import label_vix_regime  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import pandas as pd
    source = pd.read_parquet(args.input) if args.input.suffix.lower() == ".parquet" else pd.read_csv(args.input)
    out = label_vix_regime(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"Wrote {len(out):,} monthly regime observations to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Pull and cache a server-filtered CRSP monthly research sample from WRDS."""
from __future__ import annotations

import argparse
from pathlib import Path

from rdsrp.data.wrds import cached_crsp_monthly, connect_wrds
from rdsrp.utils.checks import crsp_monthly_validation_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2018-01-01", help="First CRSP calendar date.")
    parser.add_argument("--end", default="2025-12-31", help="Last CRSP calendar date.")
    parser.add_argument("--ticker", default=None, help="Optional ticker filter for diagnostic pulls.")
    parser.add_argument("--min-price", type=float, default=None, help="Optional price filter in dollars.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for diagnostics only.")
    parser.add_argument("--refresh", action="store_true", help="Ignore an existing local cache.")
    parser.add_argument(
        "--cache-name",
        default="crsp_monthly_quick.parquet",
        help="Filename written under data/raw/.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    cache_path = repo_root / "data" / "raw" / args.cache_name

    if cache_path.exists() and not args.refresh:
        from rdsrp.data.cache import read_parquet_cache

        print("[1/4] Existing CRSP cache found; WRDS connection is not required.")
        data = read_parquet_cache(cache_path)
        print(f"[2/4] Loaded cache: {cache_path}")
        print(f"[3/4] Rows={len(data):,}; stocks={data['permno'].nunique():,}; dates={data['date'].min()} to {data['date'].max()}")
        validation_path = repo_root / "data" / "interim" / f"{Path(args.cache_name).stem}_validation.csv"
        crsp_monthly_validation_report(data).to_csv(validation_path, index=False)
        print(f"[4/4] Validation report: {validation_path}")
        return 0

    db = None
    try:
        print("[1/4] Connecting to WRDS...")
        db = connect_wrds()
        data, source, _ = cached_crsp_monthly(
            db,
            cache_path=cache_path,
            start_date=args.start,
            end_date=args.end,
            ticker=args.ticker,
            min_price=args.min_price,
            row_limit=args.limit,
            refresh=args.refresh,
        )
        print(f"[2/4] CRSP source: {source.library}.{source.table} ({source.format.upper()})")
        print(f"[3/4] Rows={len(data):,}; stocks={data['permno'].nunique():,}; dates={data['date'].min()} to {data['date'].max()}")
        validation_path = repo_root / "data" / "interim" / f"{Path(args.cache_name).stem}_validation.csv"
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        crsp_monthly_validation_report(data).to_csv(validation_path, index=False)
        print(f"[4/4] Cached real WRDS data: {cache_path}")
        print(f"      Validation report: {validation_path}")
        return 0
    except Exception as exc:
        print(f"CRSP pull failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

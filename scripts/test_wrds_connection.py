"""Validate WRDS authentication and discover current CRSP and Cboe VIX sources."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdsrp.data.vix import pull_vix_daily, resolve_vix_source  # noqa: E402
from rdsrp.data.wrds import connect_wrds, pull_crsp_monthly, resolve_crsp_monthly_source  # noqa: E402


def main() -> int:
    print("[1/5] Connecting to WRDS...")
    db = None
    try:
        db = connect_wrds()
        print("WRDS connection successful.")

        print("[2/5] Discovering CRSP monthly stock source...")
        source = resolve_crsp_monthly_source(db)
        print(f"Resolved CRSP source: {source.library}.{source.table} ({source.format.upper()})")
        if source.format == "ciz":
            print("Current CIZ format detected.")
        else:
            print("Legacy CRSP format detected; current CIZ access was not found for this account.")

        print("[3/5] Pulling a tiny CRSP common-stock sample...")
        sample, _ = pull_crsp_monthly(
            db,
            start_date="2024-01-01",
            end_date="2024-03-31",
            row_limit=8,
            source=source,
        )
        if sample.empty:
            raise RuntimeError("CRSP source resolved, but the tiny validation query returned zero rows.")
        display_cols = [
            col
            for col in ["permno", "ticker", "date", "ret", "prc", "shrout", "market_equity"]
            if col in sample.columns
        ]
        print(sample[display_cols].head(8).to_string(index=False))

        print("[4/5] Discovering and sampling the Cboe VIX source...")
        vix_source = resolve_vix_source(db)
        print(
            "Resolved VIX source: "
            f"{vix_source.library}.{vix_source.table} "
            f"({vix_source.date_col}, {vix_source.value_col})"
        )
        vix, _ = pull_vix_daily(db, "2024-01-01", "2024-01-10", source=vix_source)
        if vix.empty:
            raise RuntimeError("VIX source resolved, but the tiny validation query returned zero rows.")
        print(vix.head(5).to_string(index=False))

        print("[5/5] Closing WRDS connection...")
        return 0
    except Exception as exc:
        print(f"WRDS validation failed: {type(exc).__name__}: {exc}")
        print(
            "Check your WRDS account access/authentication. Do not place a WRDS password in source code, YAML, or chat."
        )
        return 1
    finally:
        if db is not None:
            try:
                db.close()
                print("WRDS connection closed cleanly.")
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

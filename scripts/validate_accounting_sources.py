"""Validate live WRDS Compustat fundamentals and CCM link-history sources."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdsrp.data.accounting import (  # noqa: E402
    COMP_FIELDS,
    CCM_FIELDS,
    pull_ccm_links,
    pull_compustat_fundamentals,
    resolve_ccm_source,
    resolve_compustat_source,
)
from rdsrp.data.wrds import connect_wrds  # noqa: E402


def main() -> int:
    print("[1/5] Connecting securely to WRDS...")
    db = None
    try:
        db = connect_wrds()
        print("WRDS connection successful.")

        print("[2/5] Resolving Compustat annual fundamentals source...")
        comp_source = resolve_compustat_source(db)
        relevant_comp = [c for c in COMP_FIELDS if c in comp_source.columns]
        print(f"Resolved Compustat source: {comp_source.library}.{comp_source.table}")
        print("Relevant Compustat fields: " + ", ".join(relevant_comp))

        print("[3/5] Pulling a tiny Compustat sample...")
        comp, _ = pull_compustat_fundamentals(
            db,
            "2023-01-01",
            "2024-12-31",
            source=comp_source,
            row_limit=5,
        )
        if comp.empty:
            raise RuntimeError("Compustat source resolved, but the validation query returned zero rows.")
        display_comp = [
            c
            for c in ["gvkey", "datadate", "fyear", "seq", "ceq", "at", "lt", "txditc", "pstkrv"]
            if c in comp.columns
        ]
        print(comp[display_comp].head(5).to_string(index=False))

        print("[4/5] Resolving and sampling CCM link history...")
        ccm_source = resolve_ccm_source(db)
        relevant_ccm = [c for c in CCM_FIELDS if c in ccm_source.columns]
        print(f"Resolved CCM source: {ccm_source.library}.{ccm_source.table}")
        print("Relevant CCM fields: " + ", ".join(relevant_ccm))
        ccm, _ = pull_ccm_links(db, source=ccm_source, row_limit=5)
        if ccm.empty:
            raise RuntimeError("CCM source resolved, but the validation query returned zero rows.")
        print(ccm[["gvkey", "lpermno", "linktype", "linkprim", "linkdt", "linkenddt"]].to_string(index=False))

        print("[5/5] Accounting source validation succeeded.")
        return 0
    except Exception as exc:
        print(f"Accounting source validation failed: {type(exc).__name__}: {exc}")
        print(
            "Check WRDS access/authentication and product entitlements. "
            "Do not place a WRDS password in source code, configuration, or chat."
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

"""Run M10 turnover and trading-cost analysis without forecasting or WRDS access."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdsrp.transaction_costs.pipeline import run_transaction_costs  # noqa: E402
from finalize_transaction_cost_docs import update_transaction_cost_docs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", action="store_true", required=True)
    parser.parse_args()
    try:
        paths = run_transaction_costs(REPO_ROOT)
        update_transaction_cost_docs(REPO_ROOT)
        # Strict terminology check is part of completion, not an optional later step.
        import subprocess
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "audit_terminology.py"), "--strict"],
                       cwd=REPO_ROOT, check=True)
    except Exception as exc:
        print(f"Transaction-cost research run FAILED (no upstream outputs modified): {exc}")
        return 1
    print("Transaction-cost analysis complete")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

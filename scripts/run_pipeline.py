"""Run the full fixed-design pipeline end-to-end.

Order (fixed):
1) data pull (WRDS CRSP monthly + VIX series)
2) feature build
3) regime labels (VIX rolling median, 2 regimes)
4) strict OOS train/predict (expanding window; tuning inside training only)
5) metrics + model comparison
6) economic value portfolios (decile 10–1; EW & VW; turnover + tx costs)
7) tables/figures for paper
"""
from __future__ import annotations

from pathlib import Path

from rdsrp.config import load_all
from rdsrp.logging_config import setup_logging


def main() -> None:
    setup_logging()

    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_all(repo_root / "configs")

    # 1) data pull → data/raw (ignored)
    # 2) feature build → data/processed (ignored)
    # 3) regime labels → data/processed (ignored)
    # 4) OOS train/predict → results/metrics + results/predictions (git-optional)
    # 5) metrics
    # 6) portfolios → results/portfolios + results/tables
    # 7) paper artifacts → paper/ or results/tables+figures

    raise SystemExit("Stub: implement pipeline orchestration here.")


if __name__ == "__main__":
    main()

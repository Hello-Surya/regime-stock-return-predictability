"""Evaluate portfolio economic value and HAC inference from saved production predictions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdsrp.baseline.economic_value import (  # noqa: E402
    monthly_decile_long_short_returns,
    portfolio_regime_difference_tests,
    rank_ic_inference,
    summarize_portfolio_performance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=50.0,
        help="One-way transaction cost in basis points per dollar of turnover (default: 50).",
    )
    parser.add_argument(
        "--hac-lags",
        type=int,
        default=6,
        help="Newey-West/HAC lag length in months (default: 6).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = REPO_ROOT / "results" / "baseline_models"
    prediction_path = out / "baseline_oos_predictions.parquet"
    rank_path = out / "monthly_rank_ic.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Production prediction artifact not found: {prediction_path}. Run the baseline production stage first."
        )
    if not rank_path.exists():
        raise FileNotFoundError(f"Monthly rank-IC artifact not found: {rank_path}.")

    print("[1/4] Loading saved production OOS predictions...")
    predictions = pd.read_parquet(prediction_path)

    print("[2/4] Constructing tie-safe EW/VW prediction-sorted decile portfolios...")
    monthly = monthly_decile_long_short_returns(
        predictions,
        transaction_cost_bps=args.transaction_cost_bps,
    )

    print("[3/4] Computing gross/net performance and HAC regime inference...")
    performance = summarize_portfolio_performance(monthly, hac_lags=args.hac_lags)
    regime_tests = portfolio_regime_difference_tests(monthly, hac_lags=args.hac_lags)
    monthly_rank = pd.read_csv(rank_path, parse_dates=["date"])
    rank_inference, rank_regime_tests = rank_ic_inference(monthly_rank, hac_lags=args.hac_lags)

    print("[4/4] Writing economic-value research artifacts...")
    monthly.to_csv(out / "portfolio_monthly_returns.csv", index=False)
    performance.to_csv(out / "portfolio_performance.csv", index=False)
    regime_tests.to_csv(out / "portfolio_regime_tests.csv", index=False)
    rank_inference.to_csv(out / "rank_ic_inference.csv", index=False)
    rank_regime_tests.to_csv(out / "rank_ic_regime_tests.csv", index=False)

    print()
    print("Portfolio performance:")
    print(
        performance[
            [
                "model",
                "weighting",
                "regime",
                "gross_mean_monthly",
                "gross_hac_t",
                "net_mean_monthly",
                "net_annualized_mean",
                "net_sharpe",
                "net_hac_t",
                "net_p_value",
            ]
        ].to_string(index=False)
    )
    print()
    print("LOW-minus-HIGH portfolio tests:")
    print(
        regime_tests[
            ["model", "weighting", "return_type", "low_minus_high", "hac_t", "p_value"]
        ].to_string(index=False)
    )
    print()
    print("Rank-IC inference:")
    print(rank_inference.to_string(index=False))
    print()
    print("LOW-minus-HIGH rank-IC tests:")
    print(rank_regime_tests.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

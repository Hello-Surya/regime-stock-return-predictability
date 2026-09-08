"""Synthetic panel generator used only for software validation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_market(n_stocks: int = 40, n_months: int = 54, start: str = "2018-01-31", random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_months < 24:
        raise ValueError("Synthetic validation requires at least 24 months.")
    rng = np.random.default_rng(random_state)
    dates = pd.date_range(start=start, periods=n_months, freq="ME")
    permnos = np.arange(10001, 10001 + n_stocks)
    vix = np.clip(18 + 4 * np.sin(np.arange(n_months) / 4.0) + rng.normal(0, 2.5, n_months), 8, None)
    vix_df = pd.DataFrame({"date": dates, "vix": vix})
    rows: list[dict[str, object]] = []
    stock_size = rng.normal(10.0, 0.8, n_stocks)
    stock_quality = rng.normal(0.0, 1.0, n_stocks)
    stock_base_price = rng.uniform(15.0, 180.0, n_stocks)
    ret_history = np.zeros((n_months, n_stocks), dtype=float)
    for t in range(n_months):
        high_vol = vix[t] > np.median(vix[: t + 1])
        market_shock = rng.normal(0.004, 0.035 if high_vol else 0.02)
        for j, permno in enumerate(permnos):
            lag1 = ret_history[t - 1, j] if t >= 1 else 0.0
            lag2 = ret_history[t - 2, j] if t >= 2 else 0.0
            nonlinear = 0.012 * np.tanh(stock_quality[j] * lag2 * 10.0) if high_vol else 0.0
            ret = market_shock + 0.05 * lag1 + nonlinear - 0.0015 * (stock_size[j] - 10) + rng.normal(0, 0.035)
            ret = float(np.clip(ret, -0.35, 0.35))
            ret_history[t, j] = ret
            price = max(5.0, stock_base_price[j] * np.prod(1 + ret_history[: t + 1, j]))
            shares = float(np.exp(stock_size[j]) * (1 + 0.001 * t))
            rows.append({"permno": int(permno), "ticker": f"STK{j + 1:03d}", "date": dates[t], "ret": ret, "prc": price, "shrout": shares, "book_equity": 0.55 * price * shares * np.exp(0.1 * stock_quality[j])})
    return pd.DataFrame(rows), vix_df

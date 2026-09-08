"""Research figures for preliminary cross-sectional portfolio evaluation."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_cumulative_long_short(long_short: pd.DataFrame, *, weighting: str, path: Path) -> None:
    view = long_short[long_short["weighting"] == weighting].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, block in view.groupby("model", sort=True):
        block = block.sort_values("formation_date").dropna(subset=["D10_minus_D1"])
        if block.empty:
            continue
        cumulative = block["D10_minus_D1"].cumsum()
        ax.plot(block["formation_date"], cumulative, label=str(model).replace("_", " ").title())
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title(f"Cumulative Arithmetic D10-D1 Return ({weighting})")
    ax.set_xlabel("Formation month")
    ax.set_ylabel("Cumulative arithmetic return (sum of monthly spreads)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_regime_long_short_comparison(summary: pd.DataFrame, *, weighting: str, path: Path) -> None:
    view = summary[(summary["weighting"] == weighting) & summary["regime"].isin(["HIGH", "LOW"])].copy()
    pivot = view.pivot(index="regime", columns="model", values="mean_monthly_D10_minus_D1").reindex(["HIGH", "LOW"])
    ax = pivot.plot(kind="bar", figsize=(8, 4.8))
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title(f"Mean Monthly D10-D1 by VIX Regime ({weighting})")
    ax.set_xlabel("Formation-month VIX regime")
    ax.set_ylabel("Mean monthly D10-D1 return")
    ax.tick_params(axis="x", rotation=0)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_decile_return_profile(decile_returns: pd.DataFrame, *, weighting: str, path: Path) -> None:
    view = decile_returns[decile_returns["weighting"] == weighting].copy()
    decile_cols = [f"D{i}" for i in range(1, 11)]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for model, block in view.groupby("model", sort=True):
        means = block[decile_cols].mean(axis=0)
        ax.plot(range(1, 11), means.to_numpy(float), marker="o", label=str(model).replace("_", " ").title())
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title(f"Average Realized Return by Predicted-Return Decile ({weighting}, Overall)")
    ax.set_xlabel("Predicted-return decile")
    ax.set_ylabel("Average realized next-month return")
    ax.set_xticks(range(1, 11))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

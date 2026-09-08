"""Research-quality figures for prediction validation."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_actual_vs_predicted(stock_predictions: pd.DataFrame, path: Path) -> None:
    wide = stock_predictions.pivot_table(index="date", columns="model", values="prediction", aggfunc="first")
    actual = stock_predictions.groupby("date")["actual"].first()
    fig, ax = plt.subplots(figsize=(10, 5)); ax.plot(actual.index, actual.values, label="Actual next-month return", linewidth=1.8)
    for col in wide.columns: ax.plot(wide.index, wide[col], label=col.replace("_", " ").title(), linewidth=1.4)
    ax.axhline(0, linewidth=0.8); ax.set_title("Actual and Out-of-Sample Predicted Next-Month Returns"); ax.set_xlabel("Prediction month"); ax.set_ylabel("Return"); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_vix_regimes(vix_labeled: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8)); ax.plot(vix_labeled["date"], vix_labeled["vix"], label="VIX"); ax.plot(vix_labeled["date"], vix_labeled["vix_expanding_median"], label="Expanding median")
    high = vix_labeled[vix_labeled["regime"] == "HIGH"]; ax.scatter(high["date"], high["vix"], s=18, label="HIGH regime"); ax.set_title("Volatility Regime Classification"); ax.set_xlabel("Month"); ax.set_ylabel("VIX"); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_regime_performance(metrics: pd.DataFrame, path: Path) -> None:
    view = metrics[metrics["regime"].isin(["HIGH", "LOW"])].copy(); pivot = view.pivot(index="regime", columns="model", values="rmse"); ax = pivot.plot(kind="bar", figsize=(8, 4.8)); ax.set_title("Out-of-Sample RMSE by Volatility Regime"); ax.set_xlabel("Regime"); ax.set_ylabel("RMSE"); ax.tick_params(axis="x", rotation=0); fig=ax.get_figure(); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)

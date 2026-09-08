from __future__ import annotations
import pandas as pd

def one_way_turnover(current_weights: pd.Series, previous_weights: pd.Series) -> float:
    aligned = pd.concat([current_weights.rename("cur"), previous_weights.rename("prev")], axis=1).fillna(0.0)
    return float(0.5 * (aligned["cur"] - aligned["prev"]).abs().sum())

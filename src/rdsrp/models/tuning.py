"""Forward-chaining utilities for time-respecting hyperparameter selection."""
from __future__ import annotations
from collections.abc import Iterator
from itertools import product
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from rdsrp.models.elastic_net import ElasticNetModel
from rdsrp.models.xgboost_model import XGBoostModel


def forward_chaining_splits(dates: pd.Series, n_splits: int = 5, min_train_months: int = 12) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    d = pd.to_datetime(dates).reset_index(drop=True)
    unique = np.array(sorted(d.dropna().unique()))
    if len(unique) <= min_train_months:
        return
    max_boundary = len(unique) - 1
    first_boundary = min(min_train_months, max_boundary)
    boundaries = np.linspace(first_boundary, max_boundary, min(n_splits, max_boundary), dtype=int)
    seen: set[int] = set()
    for boundary in boundaries:
        boundary = int(boundary)
        if boundary in seen or boundary <= 0:
            continue
        seen.add(boundary)
        train_dates = unique[:boundary]; valid_date = unique[boundary]
        train_idx = np.flatnonzero(d.isin(train_dates).to_numpy()); valid_idx = np.flatnonzero((d == valid_date).to_numpy())
        if len(train_idx) and len(valid_idx):
            yield train_idx, valid_idx


def _grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not param_grid: return [{}]
    keys = list(param_grid)
    return [dict(zip(keys, values, strict=True)) for values in product(*(param_grid[k] for k in keys))]


def _model(model_name: str, params: dict[str, Any], random_state: int):
    if model_name == "elastic_net": return ElasticNetModel(params=params, random_state=random_state)
    if model_name == "xgboost": return XGBoostModel(params=params, random_state=random_state)
    raise ValueError(f"Unsupported model for tuning: {model_name}")


def select_params_time_cv(train: pd.DataFrame, feature_cols: list[str], model_name: str, param_grid: dict[str, list[Any]], *, target_col: str = "next_month_return", date_col: str = "date", n_splits: int = 3, min_train_months: int = 24, random_state: int = 42) -> tuple[dict[str, Any], pd.DataFrame]:
    required = {date_col, target_col, *feature_cols}; missing = required.difference(train.columns)
    if missing: raise ValueError(f"Tuning data missing columns: {sorted(missing)}")
    frame = train.dropna(subset=[target_col]).copy().sort_values(date_col).reset_index(drop=True)
    splits = list(forward_chaining_splits(frame[date_col], n_splits=n_splits, min_train_months=min_train_months))
    if not splits: raise ValueError("Insufficient historical months for forward-chaining parameter selection.")
    rows: list[dict[str, Any]] = []
    for params in _grid(param_grid):
        fold_mse: list[float] = []
        for train_idx, valid_idx in splits:
            tr = frame.iloc[train_idx]; va = frame.iloc[valid_idx]
            if tr[date_col].max() >= va[date_col].min(): raise AssertionError("Time-CV leakage guard failed.")
            model = _model(model_name, params, random_state)
            fit = model.fit(tr[feature_cols].to_numpy(float), tr[target_col].to_numpy(float))
            pred = model.predict(fit, va[feature_cols].to_numpy(float))
            fold_mse.append(float(mean_squared_error(va[target_col].to_numpy(float), pred)))
        rows.append({"model": model_name, "params": params, "mean_cv_mse": float(np.mean(fold_mse)), "n_folds": len(fold_mse)})
    results = pd.DataFrame(rows).sort_values("mean_cv_mse").reset_index(drop=True)
    return dict(results.iloc[0]["params"]), results

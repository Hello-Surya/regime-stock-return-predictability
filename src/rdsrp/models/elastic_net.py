"""Elastic Net regularized linear benchmark without model-stage imputation."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import FitResult, ModelSpec


def _finite_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} contains missing or non-finite predictors; baseline models require complete-case inputs and do not impute."
        )
    return array


class ElasticNetModel(ModelSpec):
    name = "elastic_net"

    def __init__(self, params: dict[str, Any] | None = None, random_state: int = 42) -> None:
        self.params = params or {}
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        x = _finite_matrix(X, name="Elastic Net training matrix")
        target = np.asarray(y, dtype=float).reshape(-1)
        if len(target) != len(x) or not np.isfinite(target).all():
            raise ValueError("Elastic Net target must be finite and aligned with the training matrix.")
        params = {
            "alpha": 0.001,
            "l1_ratio": 0.5,
            "max_iter": 10000,
            "random_state": self.random_state,
            **self.params,
        }
        estimator = Pipeline(
            [("scaler", StandardScaler()), ("model", ElasticNet(**params))]
        )
        estimator.fit(x, target)
        return FitResult(model_name=self.name, estimator=estimator, params=params)

    def predict(self, fit: FitResult, X: np.ndarray) -> np.ndarray:
        x = _finite_matrix(X, name="Elastic Net prediction matrix")
        return np.asarray(fit.estimator.predict(x), dtype=float)

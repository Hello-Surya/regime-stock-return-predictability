"""Elastic Net regularized linear benchmark."""
from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .base import FitResult, ModelSpec

class ElasticNetModel(ModelSpec):
    name = "elastic_net"
    def __init__(self, params: dict[str, Any] | None = None, random_state: int = 42) -> None:
        self.params = params or {}
        self.random_state = random_state
    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        params = {"alpha": 0.001, "l1_ratio": 0.5, "max_iter": 10000, "random_state": self.random_state, **self.params}
        estimator = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", ElasticNet(**params))])
        estimator.fit(X, y)
        return FitResult(model_name=self.name, estimator=estimator, params=params)

"""XGBoost nonlinear return-prediction model."""
from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from .base import FitResult, ModelSpec

class XGBoostModel(ModelSpec):
    name = "xgboost"
    def __init__(self, params: dict[str, Any] | None = None, random_state: int = 42) -> None:
        self.params = params or {}
        self.random_state = random_state
    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        params = {"objective": "reg:squarederror", "n_estimators": 100, "max_depth": 2, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": self.random_state, "n_jobs": 1, "verbosity": 0, **self.params}
        estimator = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", XGBRegressor(**params))])
        estimator.fit(X, y)
        return FitResult(model_name=self.name, estimator=estimator, params=params)

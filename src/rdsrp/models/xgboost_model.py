"""XGBoost model (only non-linear model allowed by design)."""
from __future__ import annotations

import numpy as np
from .base import FitResult, ModelSpec


class XGBoostModel(ModelSpec):
    """XGBoost with time-respecting CV inside training only."""
    name = "xgboost"

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        # Stub
        return FitResult(model_name=self.name, params={})

    def predict(self, fit: FitResult, X: np.ndarray) -> np.ndarray:
        # Stub
        return np.zeros(X.shape[0])

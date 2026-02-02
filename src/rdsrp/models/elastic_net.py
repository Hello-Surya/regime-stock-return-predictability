"""Elastic Net benchmark model."""
from __future__ import annotations

import numpy as np
from .base import FitResult, ModelSpec


class ElasticNetModel(ModelSpec):
    """Elastic Net with time-respecting CV inside training only."""
    name = "elastic_net"

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        # Stub
        return FitResult(model_name=self.name, params={})

    def predict(self, fit: FitResult, X: np.ndarray) -> np.ndarray:
        # Stub
        return np.zeros(X.shape[0])

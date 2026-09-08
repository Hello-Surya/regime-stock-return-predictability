"""Model interfaces used by the expanding-window experiment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class FitResult:
    model_name: str
    estimator: Any
    params: dict[str, Any]

class ModelSpec:
    name: str
    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        raise NotImplementedError
    def predict(self, fit: FitResult, X: np.ndarray) -> np.ndarray:
        return np.asarray(fit.estimator.predict(X), dtype=float)

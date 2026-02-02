"""Model interface for time-respecting training and prediction."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class FitResult:
    """Holds a fitted model and metadata."""
    model_name: str
    params: dict


class ModelSpec:
    """Abstract interface for models used in the paper."""
    name: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        """Fit on training data."""
        raise NotImplementedError

    def predict(self, fit: FitResult, X: np.ndarray) -> np.ndarray:
        """Predict expected returns."""
        raise NotImplementedError

"""Time-respecting hyperparameter tuning.

All tuning must occur strictly within training windows using time-based CV.
No peeking into test months.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np


def time_cv_search(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, Any],
) -> Tuple[Dict[str, Any], float]:
    """Return best_params and best_score from time-respecting CV."""
    # Stub
    return {}, float("nan")

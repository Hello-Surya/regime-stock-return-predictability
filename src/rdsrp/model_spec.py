"""Central contract for the frozen baseline model inputs."""
from __future__ import annotations

BASELINE_FEATURES: tuple[str, ...] = (
    "log_me",
    "book_to_market",
    "mom_12_2",
)
BASELINE_TARGET = "next_month_return"
FORMATION_DATE = "formation_date"
REALIZED_RETURN_DATE = "realized_return_date"

__all__ = [
    "BASELINE_FEATURES",
    "BASELINE_TARGET",
    "FORMATION_DATE",
    "REALIZED_RETURN_DATE",
]

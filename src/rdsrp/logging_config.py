"""Central logging configuration.

Keeps logs consistent across scripts and makes pipeline runs auditable.
"""
from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    """Initialize root logger with a reproducible format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

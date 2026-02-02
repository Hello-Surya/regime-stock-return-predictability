"""Cache policy for derived datasets.

This module defines canonical cache locations (raw/interim/processed).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    """Local data folders (ignored by git)."""
    raw: Path
    interim: Path
    processed: Path

"""Generate paper-ready figures from results/ (PDF/PNG)."""
from __future__ import annotations

from pathlib import Path


def build_figures(results_dir: Path, out_dir: Path) -> None:
    """Create figures consumed by paper/main.tex."""
    # Stub
    out_dir.mkdir(parents=True, exist_ok=True)

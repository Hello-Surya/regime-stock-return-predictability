"""Generate paper-ready tables from results/ (LaTeX or CSV)."""
from __future__ import annotations

from pathlib import Path


def build_tables(results_dir: Path, out_dir: Path) -> None:
    """Create LaTeX tables consumed by paper/main.tex."""
    # Stub
    out_dir.mkdir(parents=True, exist_ok=True)

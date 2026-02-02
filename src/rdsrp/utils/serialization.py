"""JSON/YAML serialization helpers for run metadata and artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def write_json(obj: Dict[str, Any], path: Path) -> None:
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")

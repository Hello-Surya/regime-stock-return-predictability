"""Configuration loading and validation.

Loads YAML configs from configs/ and enforces the fixed research design.
Any deviation should raise a ValueError to prevent accidental scope creep.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML as a plain dict."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_design(cfg: Dict[str, Any]) -> None:
    """Validate the fixed design constraints; raise on mismatch."""
    # Stub: implement strict checks in real code.
    return


@dataclass(frozen=True)
class ConfigBundle:
    """Convenience container for all configs."""
    universe: Dict[str, Any]
    regime: Dict[str, Any]
    models: Dict[str, Any]
    eval: Dict[str, Any]
    paths: Dict[str, Any]


def load_all(config_dir: Path) -> ConfigBundle:
    """Load and validate all YAML configs from config_dir."""
    universe = load_yaml(config_dir / "universe.yaml")
    regime = load_yaml(config_dir / "regime.yaml")
    models = load_yaml(config_dir / "models.yaml")
    eval_cfg = load_yaml(config_dir / "eval.yaml")
    paths = load_yaml(config_dir / "paths.yaml")

    bundle = {
        "universe": universe,
        "regime": regime,
        "models": models,
        "eval": eval_cfg,
        "paths": paths,
    }
    validate_design(bundle)
    return ConfigBundle(**bundle)

"""Configuration loading and baseline-design validation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return data


def validate_design(cfg: dict[str, Any]) -> None:
    universe = cfg["universe"]
    regime = cfg["regime"]
    models = cfg["models"]
    evaluation = cfg["eval"]

    if universe["universe"]["frequency"] != "monthly":
        raise ValueError("Baseline design requires monthly observations.")
    if universe["universe"]["shrcd"] != [10, 11]:
        raise ValueError("Baseline design requires CRSP common-share codes 10 and 11.")
    if universe["target"]["horizon_months"] != 1:
        raise ValueError("Target horizon must remain one month.")
    if regime["regime"]["method"] != "rolling_median":
        raise ValueError("Baseline regime method must be the expanding VIX median.")
    if regime["regime"]["lookback"] != "expanding":
        raise ValueError("VIX threshold must use expanding historical information.")
    if set(models["models"]["allowed"]) != {"elastic_net", "xgboost"}:
        raise ValueError("Baseline models must be Elastic Net and XGBoost only.")
    if evaluation["evaluation"]["scheme"] != "expanding_window_oos":
        raise ValueError("Evaluation must use an expanding-window OOS design.")


@dataclass(frozen=True)
class ConfigBundle:
    universe: dict[str, Any]
    regime: dict[str, Any]
    models: dict[str, Any]
    eval: dict[str, Any]
    paths: dict[str, Any]


def load_all(config_dir: Path) -> ConfigBundle:
    bundle = {
        "universe": load_yaml(config_dir / "universe.yaml"),
        "regime": load_yaml(config_dir / "regime.yaml"),
        "models": load_yaml(config_dir / "models.yaml"),
        "eval": load_yaml(config_dir / "eval.yaml"),
        "paths": load_yaml(config_dir / "paths.yaml"),
    }
    validate_design(bundle)
    return ConfigBundle(**bundle)

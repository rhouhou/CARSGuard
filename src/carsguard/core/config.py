from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .exceptions import ConfigurationError


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in config file: {path}") from exc

    if not isinstance(data, dict):
        raise ConfigurationError(f"Top-level YAML object must be a dictionary: {path}")

    return data


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.
    """
    merged = dict(base)

    for key, value in updates.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value

    return merged


def load_project_configs(config_dir: str | Path = "configs") -> Dict[str, Any]:
    """
    Load and merge all standard project config files.
    """
    config_dir = Path(config_dir)

    default_cfg = load_yaml_config(config_dir / "default.yaml")
    preprocessing_cfg = load_yaml_config(config_dir / "preprocessing.yaml")
    references_cfg = load_yaml_config(config_dir / "references.yaml")
    scoring_cfg = load_yaml_config(config_dir / "scoring.yaml")

    merged = deep_update(default_cfg, {"preprocessing": preprocessing_cfg})
    merged = deep_update(merged, {"references": references_cfg})
    merged = deep_update(merged, {"scoring": scoring_cfg})

    return merged
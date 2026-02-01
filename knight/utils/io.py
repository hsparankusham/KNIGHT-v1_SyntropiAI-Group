"""I/O utilities for loading configs, data, and checkpoints."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(data: dict[str, Any], path: Path) -> None:
    """Save data to YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: Path, indent: int = 2) -> None:
    """Save data to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist, return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load project config, merging defaults with overrides.

    Args:
        config_path: Path to override config. If None, loads defaults only.

    Returns:
        Merged configuration dict.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    defaults_path = project_root / "config" / "defaults.yaml"

    if defaults_path.exists():
        config = load_yaml(defaults_path)
    else:
        logger.warning("No defaults.yaml found at %s", defaults_path)
        config = {}

    if config_path is not None and config_path.exists():
        overrides = load_yaml(config_path)
        config = _deep_merge(config, overrides)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

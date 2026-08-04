"""Configuration loading, recursive merging, and stable hashing."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import TypeAlias, cast

import yaml

ConfigValue: TypeAlias = (
    None | bool | int | float | str | list["ConfigValue"] | dict[str, "ConfigValue"]
)
Config: TypeAlias = dict[str, ConfigValue]


def _read_yaml(path: Path) -> Config:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return cast(Config, value)


def merge_configs(base: Config, override: Config) -> Config:
    """Recursively merge an override onto a base configuration."""

    merged = copy.deepcopy(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = merge_configs(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def hash_config(config: Config) -> str:
    """Return a stable SHA-256 hash of a resolved configuration."""

    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_config(base_path: Path, condition_path: Path) -> tuple[Config, str]:
    """Load and resolve base plus condition YAML, returning config and hash."""

    resolved = merge_configs(_read_yaml(base_path), _read_yaml(condition_path))
    return resolved, hash_config(resolved)

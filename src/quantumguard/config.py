from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


class Config(dict):
    """Dict-like config that also allows attribute access, recursively."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def path(self, key: str) -> Path:
        """Resolve a paths.* config entry relative to the repo root."""
        return REPO_ROOT / self["paths"][key]


@lru_cache(maxsize=None)
def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(raw)

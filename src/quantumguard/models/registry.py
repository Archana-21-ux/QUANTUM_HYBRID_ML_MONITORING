from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quantumguard.config import Config, load_config


def _registry_path(config: Config | None = None) -> Path:
    cfg = config if config is not None else load_config()
    return cfg.path("registry_path")


def load_registry(config: Config | None = None) -> dict[str, Any]:
    path = _registry_path(config)
    if not path.exists():
        return {"models": []}
    with open(path, "r") as f:
        return json.load(f)


def _save_registry(registry: dict[str, Any], config: Config | None = None) -> None:
    path = _registry_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def register_model(
    *,
    version: str,
    model_filename: str,
    metrics: dict[str, Any],
    trained_on: str,
    deployed: bool = False,
    config: Config | None = None,
) -> dict[str, Any]:
    """Append a model version entry to registry.json and return the entry."""
    registry = load_registry(config)
    entry = {
        "version": version,
        "model_filename": model_filename,
        "metrics": metrics,
        "trained_on": trained_on,
        "deployed": deployed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    registry["models"] = [m for m in registry["models"] if m["version"] != version]
    registry["models"].append(entry)
    _save_registry(registry, config)
    return entry


def next_version(config: Config | None = None) -> str:
    """v{N} with N = 1 + the highest registered version number."""
    registry = load_registry(config)
    numbers = [
        int(match.group(1))
        for m in registry["models"]
        if (match := re.fullmatch(r"v(\d+)", m["version"]))
    ]
    return f"v{max(numbers, default=0) + 1}"


def get_latest_model_entry(config: Config | None = None) -> dict[str, Any] | None:
    registry = load_registry(config)
    if not registry["models"]:
        return None
    return registry["models"][-1]


def get_deployed_entry(config: Config | None = None) -> dict[str, Any]:
    """The entry flagged deployed; falls back to the first registered model
    (the original baseline) when nothing has been explicitly deployed yet."""
    registry = load_registry(config)
    if not registry["models"]:
        raise FileNotFoundError("no models registered yet")
    for entry in registry["models"]:
        if entry.get("deployed"):
            return entry
    return registry["models"][0]


def mark_deployed(version: str, config: Config | None = None) -> dict[str, Any]:
    """Flag `version` as the single deployed model (admin approval, Phase 6).

    Every deployment is also appended to deployment_history, which feeds the
    sliding-window model-comparison buffer."""
    registry = load_registry(config)
    target = None
    for entry in registry["models"]:
        entry["deployed"] = entry["version"] == version
        if entry["deployed"]:
            target = entry
    if target is None:
        raise FileNotFoundError(f"model version {version!r} not found in registry")
    registry.setdefault("deployment_history", []).append(version)
    _save_registry(registry, config)
    return target


def get_recent_deployments(
    limit: int, config: Config | None = None, exclude: str | None = None
) -> list[str]:
    """The rolling comparison buffer: up to `limit` most recently deployed
    versions (deduplicated, most recent first), excluding `exclude` (normally
    the currently deployed version) — older deployments simply fall off."""
    registry = load_registry(config)
    known = {entry["version"] for entry in registry["models"]}
    buffer: list[str] = []
    for version in reversed(registry.get("deployment_history", [])):
        if version == exclude or version in buffer or version not in known:
            continue
        buffer.append(version)
        if len(buffer) >= limit:
            break
    return buffer


def get_model_path(version: str | None = None, config: Config | None = None) -> Path:
    cfg = config if config is not None else load_config()
    registry = load_registry(cfg)
    if not registry["models"]:
        raise FileNotFoundError("no models registered yet")
    if version is None:
        entry = registry["models"][-1]
    else:
        matches = [m for m in registry["models"] if m["version"] == version]
        if not matches:
            raise FileNotFoundError(f"model version {version!r} not found in registry")
        entry = matches[0]
    return cfg.path("model_dir") / entry["model_filename"]

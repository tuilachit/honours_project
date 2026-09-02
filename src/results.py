"""Result persistence with mandatory reproducibility metadata."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from src.config import Config, hash_config
from src.hashing import sha256_file
from src.types import RunMetadata


def _dataset_manifest_hashes(
    config: Config,
    repository: Path,
    explicit_paths: Mapping[str, Path] | None,
) -> dict[str, str]:
    """Resolve and hash every dataset manifest named by a main experiment config."""

    datasets = config.get("datasets")
    hashes: dict[str, str] = {}
    configured_paths: dict[str, Path] = dict(explicit_paths or {})
    if isinstance(datasets, dict):
        for dataset_name, profile in datasets.items():
            if not isinstance(profile, dict):
                continue
            configured_path = profile.get("manifest_path")
            if isinstance(configured_path, str):
                configured_paths[dataset_name] = Path(configured_path)

    for dataset_name, manifest_path in configured_paths.items():
        if not manifest_path.is_absolute():
            manifest_path = repository / manifest_path
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Dataset manifest does not exist: {manifest_path}")
        hashes[dataset_name] = sha256_file(manifest_path)
    return hashes


def _json_default(value: object) -> object:
    """Serialize domain dataclasses and exact decimal values deterministically."""

    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _git_output(repository: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout


def _git_worktree_fingerprint(repository: Path, status: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(b"status\0")
    digest.update(status)
    digest.update(b"diff\0")
    digest.update(_git_output(repository, "diff", "--binary", "HEAD", "--"))
    untracked = _git_output(repository, "ls-files", "--others", "--exclude-standard", "-z")
    for raw_path in sorted(path for path in untracked.split(b"\0") if path):
        path = repository / raw_path.decode("utf-8", errors="surrogateescape")
        digest.update(b"untracked\0")
        digest.update(raw_path)
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def create_run_metadata(config: Config, repository: Path) -> RunMetadata:
    """Create provenance for a run from its config and repository state."""

    commit = _git_output(repository, "rev-parse", "HEAD").decode("ascii").strip()
    status = _git_output(repository, "status", "--porcelain=v1", "-z")
    dirty = bool(status)
    return RunMetadata(
        git_commit=commit,
        git_dirty=dirty,
        git_worktree_sha256=(
            _git_worktree_fingerprint(repository, status) if dirty else None
        ),
        config_hash=hash_config(config),
        timestamp=datetime.now(UTC).isoformat(),
    )


def write_result_json(
    path: Path,
    payload: Any,
    *,
    resolved_config: Config,
    repository: Path,
    dataset_manifest_paths: Mapping[str, Path] | None = None,
) -> RunMetadata:
    """Write a JSON result envelope containing data and complete provenance."""

    metadata = create_run_metadata(resolved_config, repository)
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "run_metadata": asdict(metadata),
        "dataset_manifest_hashes": _dataset_manifest_hashes(
            resolved_config,
            repository,
            dataset_manifest_paths,
        ),
        "resolved_config": resolved_config,
        "result": payload,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            envelope,
            handle,
            default=_json_default,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        handle.write("\n")
    return metadata

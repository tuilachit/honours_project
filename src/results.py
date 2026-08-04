"""Result persistence with mandatory reproducibility metadata."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import Config, hash_config
from src.types import RunMetadata


def create_run_metadata(config: Config, repository: Path) -> RunMetadata:
    """Create provenance for a run from its config and repository state."""

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return RunMetadata(
        git_commit=commit,
        config_hash=hash_config(config),
        timestamp=datetime.now(UTC).isoformat(),
    )


def write_result_json(
    path: Path,
    payload: Any,
    *,
    resolved_config: Config,
    repository: Path,
) -> RunMetadata:
    """Write a JSON result envelope containing data and complete provenance."""

    metadata = create_run_metadata(resolved_config, repository)
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "run_metadata": asdict(metadata),
        "resolved_config": resolved_config,
        "result": payload,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(envelope, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    return metadata

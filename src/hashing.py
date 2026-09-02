"""Shared deterministic hashing helpers for artifacts and stable identities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Hash a file incrementally without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_mapping(payload: Mapping[str, object]) -> str:
    """Hash a mapping using the project's canonical JSON encoding."""

    return sha256_json(payload)


def sha256_json(payload: object) -> str:
    """Hash a JSON-compatible value using the project's canonical encoding."""

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

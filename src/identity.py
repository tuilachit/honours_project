"""Stable identities for source cells and derived financial facts."""

from __future__ import annotations

from collections.abc import Mapping

from src.hashing import sha256_mapping


def stable_cell_id(
    *,
    dataset_revision: str,
    subset: str,
    manifest_split: str,
    context_id: str,
    table_index: int,
    row_index: int,
    column_index: int,
) -> str:
    """Return the SHA-256 identity of one semantic source-cell address."""

    if min(table_index, row_index, column_index) < 0:
        raise ValueError("Cell-address indices cannot be negative")
    return sha256_mapping(
        {
            "column_index": column_index,
            "context_id": context_id,
            "dataset_revision": dataset_revision,
            "manifest_split": manifest_split,
            "row_index": row_index,
            "subset": subset,
            "table_index": table_index,
        }
    )


def stable_fact_id(*, schema_version: str, payload: Mapping[str, object]) -> str:
    """Return a versioned identity for one derived fact representation."""

    if not schema_version:
        raise ValueError("Fact schema version cannot be empty")
    return sha256_mapping({"schema_version": schema_version, "fact": dict(payload)})

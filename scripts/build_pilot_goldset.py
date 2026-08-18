"""Validate and materialise the human-reviewed exact-cell feasibility pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import yaml

from scripts.audit_t2_ragbench import (
    SEPARATOR_CELL_PATTERN,
    extract_markdown_tables,
    sha256_file,
)
from src.config import Config, load_config
from src.results import write_result_json

TableRows = tuple[tuple[str, ...], ...]
TableBlocks = tuple[TableRows, ...]
SourceKey = tuple[str, str]


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return cast(dict[str, Any], value)


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _integer(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _boolean(mapping: dict[str, Any], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    return _mapping(value, str(path))


def _repository_path(repository: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else repository / path


def _is_separator_row(cells: Sequence[str]) -> bool:
    nonempty = [cell for cell in cells if cell]
    return bool(nonempty) and all(SEPARATOR_CELL_PATTERN.fullmatch(cell) for cell in nonempty)


def parse_table_blocks(
    text: str,
    *,
    min_pipe_count: int,
    drop_leading_synthetic_column: bool,
) -> TableBlocks:
    """Parse addressable Markdown rows without discarding blank source cells."""

    parsed_blocks: list[TableRows] = []
    for block in extract_markdown_tables(text, min_pipe_count=min_pipe_count):
        rows: list[tuple[str, ...]] = []
        for line in block:
            cells = tuple(cell.strip() for cell in line.strip("|").split("|"))
            if _is_separator_row(cells):
                continue
            if drop_leading_synthetic_column:
                if not cells:
                    raise ValueError("Cannot remove a synthetic column from an empty row")
                cells = cells[1:]
            rows.append(cells)
        parsed_blocks.append(tuple(rows))
    return tuple(parsed_blocks)


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
    """Return a stable SHA-256 identity for one semantic source-cell address."""

    if min(table_index, row_index, column_index) < 0:
        raise ValueError("Cell-address indices cannot be negative")
    address = {
        "column_index": column_index,
        "context_id": context_id,
        "dataset_revision": dataset_revision,
        "manifest_split": manifest_split,
        "row_index": row_index,
        "subset": subset,
        "table_index": table_index,
    }
    canonical = json.dumps(address, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cell_value(
    blocks: TableBlocks,
    *,
    table_index: int,
    row_index: int,
    column_index: int,
    label: str,
) -> str:
    try:
        table = blocks[table_index]
        row = table[row_index]
        return row[column_index]
    except IndexError as error:
        raise ValueError(
            f"{label} points outside the parsed table: "
            f"table={table_index}, row={row_index}, column={column_index}"
        ) from error


def _address(cell: dict[str, Any], *, default_table_index: int) -> tuple[int, int, int]:
    table_index_value = cell.get("table_index", default_table_index)
    if not isinstance(table_index_value, int) or isinstance(table_index_value, bool):
        raise ValueError("table_index must be an integer")
    address = (
        table_index_value,
        _integer(cell, "row_index"),
        _integer(cell, "column_index"),
    )
    if min(address) < 0:
        raise ValueError("Cell-address indices cannot be negative")
    return address


def _source_profiles(dataset_manifest: dict[str, Any]) -> dict[SourceKey, dict[str, Any]]:
    dataset = _mapping(dataset_manifest.get("dataset"), "dataset manifest.dataset")
    profiles: dict[SourceKey, dict[str, Any]] = {}
    for item in _list(dataset.get("files"), "dataset manifest files"):
        profile = _mapping(item, "dataset source profile")
        key = (_string(profile, "subset"), _string(profile, "split"))
        if key in profiles:
            raise ValueError(f"Duplicate dataset source profile: {key}")
        profiles[key] = profile
    return profiles


def _load_requested_records(
    *,
    repository: Path,
    profiles: dict[SourceKey, dict[str, Any]],
    requested_ids: dict[SourceKey, set[str]],
    id_field: str,
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], list[dict[str, Any]]]:
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    integrity: list[dict[str, Any]] = []
    for source_key, question_ids in sorted(requested_ids.items()):
        profile = profiles.get(source_key)
        if profile is None:
            raise ValueError(f"No manifest source for annotation subset/split: {source_key}")
        relative_path = _string(profile, "path")
        path = _repository_path(repository, relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing pilot source file: {path}")

        expected_bytes = _integer(profile, "expected_bytes")
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Byte-count mismatch for {relative_path}: {actual_bytes} != {expected_bytes}"
            )
        expected_sha256 = _string(profile, "sha256")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"SHA-256 mismatch for {relative_path}")

        remaining = set(question_ids)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                parsed = json.loads(line)
                row = _mapping(parsed, f"{relative_path}:{line_number}")
                question_id = row.get(id_field)
                if not isinstance(question_id, str) or question_id not in remaining:
                    continue
                records[(source_key[0], source_key[1], question_id)] = row
                remaining.remove(question_id)
                if not remaining:
                    break
        if remaining:
            raise ValueError(f"Missing annotated records in {relative_path}: {sorted(remaining)}")
        integrity.append(
            {
                "subset": source_key[0],
                "split": source_key[1],
                "path": relative_path,
                "bytes": actual_bytes,
                "sha256": actual_sha256,
                "verified": True,
            }
        )
    return records, integrity


def validate_pilot_annotations(config: Config, repository: Path) -> dict[str, Any]:
    """Validate approved annotations against pinned source rows and exact cells."""

    pilot = _mapping(config.get("pilot"), "pilot")
    audit = _mapping(config.get("audit"), "audit")
    datasets = _mapping(config.get("datasets"), "datasets")
    primary = _mapping(datasets.get("primary"), "datasets.primary")
    field_names = _mapping(audit.get("field_names"), "audit.field_names")

    annotation_path = _repository_path(repository, _string(pilot, "annotation_path"))
    manifest_path = _repository_path(repository, _string(primary, "manifest_path"))
    annotations_root = _load_yaml_mapping(annotation_path)
    manifest = _load_yaml_mapping(manifest_path)
    manifest_dataset = _mapping(manifest.get("dataset"), "dataset manifest.dataset")
    annotation_provenance = _mapping(
        annotations_root.get("annotation_provenance"), "annotation provenance"
    )
    selection_provenance = _mapping(
        annotations_root.get("selection_provenance"), "selection provenance"
    )
    source_review_round = _mapping(
        annotations_root.get("source_review_round"), "source review round"
    )
    human_source_review_round = _mapping(
        annotations_root.get("human_source_review_round"), "human source review round"
    )

    expected_schema_version = _integer(pilot, "expected_schema_version")
    if annotations_root.get("schema_version") != expected_schema_version:
        raise ValueError("Pilot annotation schema version does not match config")
    if annotations_root.get("status") != _string(pilot, "required_annotation_status"):
        raise ValueError("Pilot annotation status does not match config")

    dataset_annotation = _mapping(annotations_root.get("dataset"), "annotation dataset")
    dataset_revision = _string(primary, "revision")
    if _string(dataset_annotation, "revision") != dataset_revision:
        raise ValueError("Annotation dataset revision does not match the experiment config")
    if _string(manifest_dataset, "revision") != dataset_revision:
        raise ValueError("Dataset manifest revision does not match the experiment config")
    manifest_sha256 = sha256_file(manifest_path)
    if _string(dataset_annotation, "manifest_sha256") != manifest_sha256:
        raise ValueError("Annotation dataset-manifest hash is stale")

    annotations = [
        _mapping(item, "pilot annotation")
        for item in _list(annotations_root.get("annotations"), "annotations")
    ]
    sample_size = _integer(pilot, "sample_size")
    if len(annotations) != sample_size:
        raise ValueError(f"Pilot has {len(annotations)} annotations; expected {sample_size}")
    if annotations_root.get("approved_included_questions") != sample_size:
        raise ValueError("Approved annotation count does not match configured sample size")

    profiles = _source_profiles(manifest)
    requested_ids: dict[SourceKey, set[str]] = defaultdict(set)
    seen_question_ids: set[str] = set()
    for annotation in annotations:
        question_id = _string(annotation, "question_id")
        if question_id in seen_question_ids:
            raise ValueError(f"Duplicate pilot question ID: {question_id}")
        seen_question_ids.add(question_id)
        source = _mapping(annotation.get("source"), f"{question_id}.source")
        source_key = (_string(source, "subset"), _string(source, "manifest_split"))
        requested_ids[source_key].add(question_id)

    def reviewed_ids(review: dict[str, Any], key: str, label: str) -> set[str]:
        values = _list(review.get(key), f"{label}.{key}")
        if not all(isinstance(value, str) for value in values):
            raise ValueError(f"{label}.{key} must contain question IDs")
        ids = {cast(str, value) for value in values}
        if len(ids) != len(values):
            raise ValueError(f"{label}.{key} contains duplicate question IDs")
        return ids

    initially_individually_reviewed_ids = reviewed_ids(
        source_review_round,
        "individually_reviewed_by_primary_researcher",
        "source_review_round",
    )
    delegated_include_ids = reviewed_ids(
        source_review_round, "delegated_include", "source_review_round"
    )
    delegated_exclude_ids = reviewed_ids(
        source_review_round, "delegated_exclude", "source_review_round"
    )
    unresolved_review_ids = reviewed_ids(source_review_round, "unresolved", "source_review_round")
    review_groups = (
        initially_individually_reviewed_ids,
        delegated_include_ids,
        delegated_exclude_ids,
        unresolved_review_ids,
    )
    for index, left in enumerate(review_groups):
        for right in review_groups[index + 1 :]:
            if left & right:
                raise ValueError("Source-review question groups must not overlap")
    reviewed_included_ids = initially_individually_reviewed_ids | delegated_include_ids
    if reviewed_included_ids != seen_question_ids:
        raise ValueError("Source review does not cover every included pilot question")
    if delegated_exclude_ids or unresolved_review_ids:
        raise ValueError("Included annotations cannot be delegated exclusions or unresolved")
    if _integer(annotation_provenance, "delegated_source_review_question_count") != len(
        delegated_include_ids
    ):
        raise ValueError("Delegated source-review count does not match its question list")
    if _boolean(annotation_provenance, "delegated_source_review_completed") is not True:
        raise ValueError("Delegated source review is not marked complete")

    human_reviewed_ids = reviewed_ids(
        human_source_review_round,
        "reviewed_question_ids",
        "human_source_review_round",
    )
    human_include_ids = reviewed_ids(
        human_source_review_round, "include", "human_source_review_round"
    )
    human_exclude_ids = reviewed_ids(
        human_source_review_round, "exclude", "human_source_review_round"
    )
    human_unsure_ids = reviewed_ids(
        human_source_review_round, "unsure", "human_source_review_round"
    )
    human_decision_groups = (human_include_ids, human_exclude_ids, human_unsure_ids)
    for index, left in enumerate(human_decision_groups):
        for right in human_decision_groups[index + 1 :]:
            if left & right:
                raise ValueError("Human source-review decision groups must not overlap")
    if human_reviewed_ids != seen_question_ids:
        raise ValueError("Human source review does not cover every pilot question")
    if human_include_ids != seen_question_ids or human_exclude_ids or human_unsure_ids:
        raise ValueError("Human source-review decisions do not match included annotations")
    if _string(human_source_review_round, "review_mode") != "human_full_source_review":
        raise ValueError("Human source-review mode is invalid")
    if _integer(annotation_provenance, "human_source_review_question_count") != len(
        human_reviewed_ids
    ):
        raise ValueError("Human source-review count does not match its question list")
    if _boolean(annotation_provenance, "human_source_review_completed") is not True:
        raise ValueError("Human source review is not marked complete")
    _string(human_source_review_round, "reviewer")
    _string(human_source_review_round, "completed_at")
    _string(human_source_review_round, "declaration")
    human_review_material = _mapping(
        human_source_review_round.get("review_material"), "human source review material"
    )
    for key in ("path", "generator_git_commit", "config_hash"):
        _string(human_review_material, key)

    records, source_integrity = _load_requested_records(
        repository=repository,
        profiles=profiles,
        requested_ids=requested_ids,
        id_field=_string(field_names, "id"),
    )

    transforms = _mapping(pilot.get("coordinate_transforms"), "coordinate transforms")
    min_pipe_count = _integer(audit, "markdown_min_pipe_count")
    minimum_valid_cells = _integer(pilot, "minimum_valid_cells_per_question")
    minimum_near_misses = _integer(pilot, "minimum_near_misses_per_question")
    required_decision = _string(pilot, "required_decision")
    required_reasoning = _string(pilot, "required_reasoning_type")
    required_review_status = _string(pilot, "required_review_status")
    if _string(pilot, "table_hash_algorithm") != "sha256":
        raise ValueError("Only SHA-256 table hashes are currently supported")
    if _string(pilot, "cell_id_hash_algorithm") != "sha256":
        raise ValueError("Only SHA-256 cell IDs are currently supported")

    subset_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    mismatch_field_counts: Counter[str] = Counter()
    gold_labels: list[dict[str, Any]] = []
    near_miss_labels: list[dict[str, Any]] = []
    gold_cell_ids: set[str] = set()
    near_miss_cell_ids: set[str] = set()
    near_miss_count = 0
    ambiguity_count = 0
    measured_annotation_seconds: list[float] = []

    for annotation in annotations:
        question_id = _string(annotation, "question_id")
        if annotation.get("decision") != required_decision:
            raise ValueError(f"{question_id} is not an included annotation")
        if annotation.get("reasoning_type") != required_reasoning:
            raise ValueError(f"{question_id} is not a direct lookup")
        review = _mapping(annotation.get("review"), f"{question_id}.review")
        if review.get("status") != required_review_status:
            raise ValueError(f"{question_id} has not been user-approved")
        if question_id in initially_individually_reviewed_ids:
            full_source_review = _mapping(
                review.get("full_source_review"), f"{question_id}.full_source_review"
            )
            if full_source_review.get("status") != "approved":
                raise ValueError(f"{question_id} full-source review is not approved")
            if full_source_review.get("decision") != required_decision:
                raise ValueError(f"{question_id} full-source decision is not include")
        annotation_seconds = review.get("annotation_seconds")
        if annotation_seconds is not None:
            if not isinstance(annotation_seconds, int | float) or isinstance(
                annotation_seconds, bool
            ):
                raise ValueError(f"{question_id}.annotation_seconds must be numeric or null")
            if annotation_seconds < 0:
                raise ValueError(f"{question_id}.annotation_seconds cannot be negative")
            measured_annotation_seconds.append(float(annotation_seconds))

        source = _mapping(annotation.get("source"), f"{question_id}.source")
        subset = _string(source, "subset")
        manifest_split = _string(source, "manifest_split")
        source_key = (subset, manifest_split)
        profile = profiles[source_key]
        record = records[(subset, manifest_split, question_id)]
        context_id = _string(source, "context_id")
        if record.get(_string(field_names, "context_id")) != context_id:
            raise ValueError(f"{question_id} context ID does not match the source row")
        if record.get(_string(field_names, "file_name")) != _string(source, "file_name"):
            raise ValueError(f"{question_id} file name does not match the source row")
        if record.get(_string(field_names, "question")) != _string(annotation, "question"):
            raise ValueError(f"{question_id} question text does not match the source row")
        if record.get(_string(field_names, "split")) != _string(source, "record_split"):
            raise ValueError(f"{question_id} record split does not match the source row")
        dataset_answer = _mapping(annotation.get("dataset_answer"), f"{question_id}.dataset_answer")
        source_program_answer = record.get(_string(field_names, "program_answer"))
        source_original_answer = record.get(_string(field_names, "original_answer"))
        if str(source_program_answer) != _string(dataset_answer, "program_raw"):
            raise ValueError(f"{question_id} program answer does not match the source row")
        if str(source_original_answer) != _string(dataset_answer, "original_raw"):
            raise ValueError(f"{question_id} original answer does not match the source row")

        configured_page_field = _string(field_names, "page_number")
        annotated_page = source.get("page_number")
        source_page = record.get(configured_page_field)
        if annotated_page is not None and str(source_page) != str(annotated_page):
            raise ValueError(f"{question_id} page number does not match the source row")

        table_text = record.get(_string(profile, "table_text_field"))
        if not isinstance(table_text, str) or not table_text:
            raise ValueError(f"{question_id} has no configured source-table text")
        raw_blocks = extract_markdown_tables(table_text, min_pipe_count=min_pipe_count)
        table_index = _integer(source, "table_index")
        try:
            canonical_table = "\n".join(raw_blocks[table_index])
        except IndexError as error:
            raise ValueError(f"{question_id} table index is outside the source context") from error
        table_hash = hashlib.sha256(canonical_table.encode("utf-8")).hexdigest()
        if table_hash != _string(source, "table_sha256"):
            raise ValueError(f"{question_id} source-table hash is stale")

        transform = _mapping(transforms.get(subset), f"coordinate transform for {subset}")
        drop_synthetic = _boolean(transform, "drop_leading_synthetic_column")
        semantic_blocks = parse_table_blocks(
            table_text,
            min_pipe_count=min_pipe_count,
            drop_leading_synthetic_column=drop_synthetic,
        )
        raw_coordinate_blocks = parse_table_blocks(
            table_text,
            min_pipe_count=min_pipe_count,
            drop_leading_synthetic_column=False,
        )

        valid_cells = [
            _mapping(item, f"{question_id}.valid_cell")
            for item in _list(annotation.get("valid_cells"), f"{question_id}.valid_cells")
        ]
        if len(valid_cells) < minimum_valid_cells:
            raise ValueError(f"{question_id} has too few valid cells")
        valid_addresses: set[tuple[int, int, int]] = set()
        valid_cell_ids: list[str] = []
        address_payloads: list[dict[str, Any]] = []
        for cell in valid_cells:
            address = _address(cell, default_table_index=table_index)
            if address in valid_addresses:
                raise ValueError(f"{question_id} contains a duplicate valid-cell address")
            valid_addresses.add(address)
            actual_value = _cell_value(
                semantic_blocks,
                table_index=address[0],
                row_index=address[1],
                column_index=address[2],
                label=f"{question_id} valid cell",
            )
            if actual_value != _string(cell, "raw_value"):
                raise ValueError(
                    f"{question_id} valid-cell value mismatch: {actual_value!r} != "
                    f"{cell.get('raw_value')!r}"
                )
            raw_column_index = cell.get("raw_markdown_column_index")
            if raw_column_index is not None:
                if not isinstance(raw_column_index, int) or isinstance(raw_column_index, bool):
                    raise ValueError(f"{question_id}.raw_markdown_column_index must be integer")
                if raw_column_index < 0:
                    raise ValueError(f"{question_id}.raw_markdown_column_index cannot be negative")
                raw_value = _cell_value(
                    raw_coordinate_blocks,
                    table_index=address[0],
                    row_index=address[1],
                    column_index=raw_column_index,
                    label=f"{question_id} raw Markdown cell",
                )
                if raw_value != actual_value:
                    raise ValueError(f"{question_id} semantic and raw coordinates disagree")
            if not _list(cell.get("raw_row_header_path"), "raw row-header path"):
                raise ValueError(f"{question_id} has an empty raw row-header path")
            if not _list(cell.get("raw_column_header_path"), "raw column-header path"):
                raise ValueError(f"{question_id} has an empty raw column-header path")

            cell_id = stable_cell_id(
                dataset_revision=dataset_revision,
                subset=subset,
                manifest_split=manifest_split,
                context_id=context_id,
                table_index=address[0],
                row_index=address[1],
                column_index=address[2],
            )
            gold_cell_ids.add(cell_id)
            valid_cell_ids.append(cell_id)
            raw_markdown_column_index = address[2] + int(drop_synthetic)
            address_payloads.append(
                {
                    "cell_id": cell_id,
                    "subset": subset,
                    "manifest_split": manifest_split,
                    "file_name": _string(source, "file_name"),
                    "page_number": source.get("page_number"),
                    "context_id": context_id,
                    "table_index": address[0],
                    "row_index": address[1],
                    "column_index": address[2],
                    "raw_markdown_column_index": raw_markdown_column_index,
                    "raw_value": actual_value,
                    "normalized_value": cell.get("normalized_value"),
                    "raw_row_header_path": _list(
                        cell.get("raw_row_header_path"), "raw row-header path"
                    ),
                    "raw_column_header_path": _list(
                        cell.get("raw_column_header_path"), "raw column-header path"
                    ),
                }
            )

        near_misses = [
            _mapping(item, f"{question_id}.near_miss")
            for item in _list(
                annotation.get("natural_near_misses"),
                f"{question_id}.natural_near_misses",
            )
        ]
        if len(near_misses) < minimum_near_misses:
            raise ValueError(f"{question_id} has too few natural near misses")
        seen_near_addresses: set[tuple[int, int, int]] = set()
        for near_miss in near_misses:
            address = _address(near_miss, default_table_index=table_index)
            if address in valid_addresses:
                raise ValueError(f"{question_id} marks a valid cell as a near miss")
            if address in seen_near_addresses:
                raise ValueError(f"{question_id} repeats a near-miss address")
            seen_near_addresses.add(address)
            actual_value = _cell_value(
                semantic_blocks,
                table_index=address[0],
                row_index=address[1],
                column_index=address[2],
                label=f"{question_id} near miss",
            )
            if actual_value != _string(near_miss, "raw_value"):
                raise ValueError(
                    f"{question_id} near-miss value mismatch: {actual_value!r} != "
                    f"{near_miss.get('raw_value')!r}"
                )
            mismatch_fields = [
                str(value)
                for value in _list(
                    near_miss.get("mismatch_fields"),
                    f"{question_id}.near_miss.mismatch_fields",
                )
            ]
            if not mismatch_fields:
                raise ValueError(f"{question_id} near miss has no mismatch field")
            mismatch_field_counts.update(mismatch_fields)
            cell_id = stable_cell_id(
                dataset_revision=dataset_revision,
                subset=subset,
                manifest_split=manifest_split,
                context_id=context_id,
                table_index=address[0],
                row_index=address[1],
                column_index=address[2],
            )
            near_miss_cell_ids.add(cell_id)
            raw_markdown_column_index = address[2] + int(drop_synthetic)
            raw_coordinate_value = _cell_value(
                raw_coordinate_blocks,
                table_index=address[0],
                row_index=address[1],
                column_index=raw_markdown_column_index,
                label=f"{question_id} raw Markdown near miss",
            )
            if raw_coordinate_value != actual_value:
                raise ValueError(f"{question_id} near-miss coordinate transform disagrees")
            near_miss_labels.append(
                {
                    "question_id": question_id,
                    "cell_id": cell_id,
                    "subset": subset,
                    "manifest_split": manifest_split,
                    "file_name": _string(source, "file_name"),
                    "page_number": source.get("page_number"),
                    "context_id": context_id,
                    "table_index": address[0],
                    "row_index": address[1],
                    "column_index": address[2],
                    "raw_markdown_column_index": raw_markdown_column_index,
                    "raw_value": actual_value,
                    "mismatch_fields": mismatch_fields,
                }
            )

        ambiguity_notes = _list(annotation.get("ambiguity_notes"), f"{question_id}.ambiguity_notes")
        ambiguity_count += bool(ambiguity_notes)
        near_miss_count += len(near_misses)
        subset_counts[subset] += 1
        split_counts[f"{subset}/{manifest_split}"] += 1
        gold_labels.append(
            {
                "question_id": question_id,
                "valid_cell_ids": valid_cell_ids,
                "addresses": address_payloads,
                "annotator_id": review.get("annotator_id"),
                "adjudicated": False,
            }
        )

    configured_targets = _mapping(pilot.get("subset_targets"), "pilot subset targets")
    expected_targets = {key: _integer(configured_targets, key) for key in configured_targets}
    if dict(sorted(subset_counts.items())) != dict(sorted(expected_targets.items())):
        raise ValueError(
            f"Pilot subset balance mismatch: {dict(subset_counts)} != {expected_targets}"
        )

    require_timing = _boolean(pilot, "require_per_question_annotation_seconds")
    if require_timing and len(measured_annotation_seconds) != sample_size:
        raise ValueError("Per-question annotation timing is required but incomplete")
    timing_ready = len(measured_annotation_seconds) == sample_size
    timing_required_for_scale = _boolean(pilot, "annotation_timing_required_for_scale_decision")
    independent_review_completed = _boolean(pilot, "independent_review_completed")
    if (
        _boolean(annotation_provenance, "independent_review_completed")
        != independent_review_completed
    ):
        raise ValueError("Annotation and config disagree about independent review status")
    complete_screening_log = _boolean(selection_provenance, "complete_screening_log_available")
    full_source_inspection = _boolean(
        annotation_provenance, "full_source_inspection_by_primary_reviewer"
    )
    if full_source_inspection != (len(human_reviewed_ids) == sample_size):
        raise ValueError("Primary full-source review status disagrees with reviewed question IDs")
    scale_decision_ready = (
        (timing_ready or not timing_required_for_scale)
        and independent_review_completed
        and complete_screening_log
        and full_source_inspection
    )
    open_gates: list[str] = []
    if not timing_ready:
        open_gates.append("active annotation time was not measured")
    if not independent_review_completed:
        open_gates.append("independent second-annotator review has not been completed")
    if not complete_screening_log:
        open_gates.append("complete candidate screening and exclusion log is unavailable")
    if not full_source_inspection:
        open_gates.append(
            "primary researcher full-source inspection is incomplete "
            f"({len(human_reviewed_ids)}/{sample_size})"
        )

    exclusions = _list(annotations_root.get("exclusions"), "exclusions")
    return {
        "pilot_id": _string(annotations_root, "pilot_id"),
        "annotation_status": _string(annotations_root, "status"),
        "validation_status": (
            "validated" if not open_gates else "validated_with_open_feasibility_gates"
        ),
        "source_integrity_and_cell_addresses_validated": True,
        "semantic_label_status": "researcher_full_source_review_complete",
        "scale_decision_ready": scale_decision_ready,
        "open_feasibility_gates": open_gates,
        "dataset": {
            "name": _string(manifest_dataset, "name"),
            "revision": dataset_revision,
            "manifest_sha256": manifest_sha256,
        },
        "annotation_provenance": annotation_provenance,
        "selection_provenance": selection_provenance,
        "source_review": {
            "mode": _string(human_source_review_round, "review_mode"),
            "primary_researcher_full_source_reviews": len(human_reviewed_ids),
            "human_includes": len(human_include_ids),
            "human_excludes": len(human_exclude_ids),
            "human_unsure": len(human_unsure_ids),
            "historical_delegated_ai_includes": len(delegated_include_ids),
            "review_material": human_review_material,
        },
        "split_policy": {
            "evaluation_use": _string(pilot, "evaluation_use"),
            "exclude_source_families_from_confirmatory_test": _boolean(
                pilot, "exclude_source_families_from_confirmatory_test"
            ),
        },
        "counts": {
            "questions": sample_size,
            "subsets": dict(sorted(subset_counts.items())),
            "source_splits": dict(sorted(split_counts.items())),
            "valid_cells": sum(len(item["valid_cell_ids"]) for item in gold_labels),
            "natural_near_misses": near_miss_count,
            "unique_gold_cell_ids": len(gold_cell_ids),
            "unique_near_miss_cell_ids": len(near_miss_cell_ids),
            "questions_with_ambiguity_notes": ambiguity_count,
            "recorded_exclusions": len(exclusions),
        },
        "mismatch_field_counts": dict(sorted(mismatch_field_counts.items())),
        "annotation_timing": {
            "measured_questions": len(measured_annotation_seconds),
            "total_questions": sample_size,
            "total_seconds": sum(measured_annotation_seconds),
            "ready_for_scale_estimate": timing_ready,
        },
        "independent_review": {
            "target_fraction": pilot.get("independent_review_target_fraction"),
            "completed": independent_review_completed,
        },
        "source_integrity": source_integrity,
        "gold_labels": gold_labels,
        "near_miss_labels": near_miss_labels,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the pilot validation command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--condition-config", required=True, type=Path)
    return parser


def build_pilot_goldset(config: Config) -> None:
    """Validate annotations and write a provenance-stamped pilot artifact."""

    repository = Path(__file__).resolve().parents[1]
    result = validate_pilot_annotations(config, repository)
    pilot = _mapping(config.get("pilot"), "pilot")
    output_path = _repository_path(repository, _string(pilot, "output_path"))
    metadata = write_result_json(
        output_path,
        result,
        resolved_config=config,
        repository=repository,
    )
    print(
        json.dumps(
            {
                "output": str(output_path.relative_to(repository)),
                "git_commit": metadata.git_commit,
                "config_hash": metadata.config_hash,
                "validation_status": result["validation_status"],
                "counts": result["counts"],
                "open_feasibility_gates": result["open_feasibility_gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve workflow configuration and validate the pilot artifact."""

    args = build_parser().parse_args(argv)
    config, _ = load_config(
        cast(Path, args.base_config),
        cast(Path, args.condition_config),
    )
    build_pilot_goldset(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

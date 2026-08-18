"""Build a human-readable source-review pack for the exact-cell pilot."""

from __future__ import annotations

import argparse
import html
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from scripts.audit_t2_ragbench import extract_markdown_tables
from scripts.build_pilot_goldset import (
    SourceKey,
    _integer,
    _list,
    _load_requested_records,
    _load_yaml_mapping,
    _mapping,
    _repository_path,
    _source_profiles,
    _string,
    validate_pilot_annotations,
)
from src.config import Config, load_config
from src.results import create_run_metadata


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _path_text(value: object) -> str:
    if not isinstance(value, list):
        return ""
    return " → ".join(str(item) for item in value)


def _address_text(cell: dict[str, Any], default_table_index: int) -> str:
    table_index = cell.get("table_index", default_table_index)
    raw_column = cell.get("raw_markdown_column_index")
    address = f"T{table_index}:r{cell.get('row_index')},c{cell.get('column_index')}"
    if raw_column is not None:
        address += f" (raw Markdown c{raw_column})"
    return address


def _source_statuses(annotations_root: dict[str, Any]) -> tuple[set[str], set[str]]:
    review_round = _mapping(annotations_root.get("source_review_round"), "source review round")
    human_review_round = _mapping(
        annotations_root.get("human_source_review_round"), "human source review round"
    )
    individual = {
        str(value)
        for value in _list(
            human_review_round.get("reviewed_question_ids"),
            "individual source reviews",
        )
    }
    delegated = {
        str(value)
        for value in _list(review_round.get("delegated_include"), "delegated source reviews")
    }
    return individual, delegated


def _render_question(
    *,
    ordinal: int,
    annotation: dict[str, Any],
    record: dict[str, Any],
    profile: dict[str, Any],
    context_field: str,
    min_pipe_count: int,
    individually_reviewed: set[str],
    delegated_reviewed: set[str],
) -> list[str]:
    question_id = _string(annotation, "question_id")
    source = _mapping(annotation.get("source"), f"{question_id}.source")
    table_index = _integer(source, "table_index")
    query = _mapping(annotation.get("query"), f"{question_id}.query")
    unit = _mapping(query.get("unit"), f"{question_id}.query.unit")
    dimensions = _list(query.get("dimensions"), f"{question_id}.query.dimensions")
    valid_cells = [
        _mapping(value, f"{question_id}.valid_cell")
        for value in _list(annotation.get("valid_cells"), f"{question_id}.valid_cells")
    ]
    near_misses = [
        _mapping(value, f"{question_id}.near_miss")
        for value in _list(
            annotation.get("natural_near_misses"), f"{question_id}.natural_near_misses"
        )
    ]

    if question_id in individually_reviewed:
        prior_status = "Previously inspected individually by the primary researcher"
    elif question_id in delegated_reviewed:
        prior_status = "Previously retained through delegated AI source review"
    else:
        prior_status = "No prior source-review status"

    source_file = _string(source, "file_name")
    page = source.get("page_number")
    page_text = "unavailable in metadata" if page is None else str(page)
    dimension_text = "; ".join(
        f"{_mapping(value, 'query dimension').get('name')}="
        f"{_mapping(value, 'query dimension').get('value')}"
        for value in dimensions
    )
    if not dimension_text:
        dimension_text = "none"

    lines = [
        f'<a id="question-{ordinal}"></a>',
        "",
        f"## {ordinal}. `{question_id}`",
        "",
        "**Decision:** ☐ INCLUDE &nbsp;&nbsp; ☐ EXCLUDE &nbsp;&nbsp; ☐ UNSURE",
        "",
        f"**Prior status:** {prior_status}",
        "",
        f"**Question:** {_string(annotation, 'question')}",
        "",
        f"**Source:** `{source_file}` · page {page_text} · "
        f"context `{_string(source, 'context_id')}`",
        "",
        "**Requested meaning:**",
        "",
        f"- Entity: {query.get('entity')}",
        f"- Concept: {query.get('concept')}",
        f"- Period: {query.get('period')}",
        f"- Unit: currency={unit.get('currency')}, scale={unit.get('scale')}, "
        f"source wording={unit.get('raw_text')}",
        f"- Other dimensions: {dimension_text}",
        "",
        "### Proposed correct cell(s)",
        "",
        "| Address | Raw value | Row-header path | Column-header path |",
        "|---|---:|---|---|",
    ]
    for cell in valid_cells:
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown_cell(_address_text(cell, table_index)),
                    _markdown_cell(cell.get("raw_value")),
                    _markdown_cell(_path_text(cell.get("raw_row_header_path"))),
                    _markdown_cell(_path_text(cell.get("raw_column_header_path"))),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "### Plausible but wrong cells",
            "",
            "| Address | Raw value | Interpreted row/column | Why wrong |",
            "|---|---:|---|---|",
        ]
    )
    for cell in near_misses:
        interpreted = " / ".join(
            str(value)
            for value in (cell.get("interpreted_row"), cell.get("interpreted_column"))
            if value is not None
        )
        mismatch_fields = ", ".join(
            str(value) for value in _list(cell.get("mismatch_fields"), "near-miss mismatch fields")
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown_cell(_address_text(cell, table_index)),
                    _markdown_cell(cell.get("raw_value")),
                    _markdown_cell(interpreted),
                    _markdown_cell(mismatch_fields),
                )
            )
            + " |"
        )

    ambiguity_notes = _list(annotation.get("ambiguity_notes"), f"{question_id}.ambiguity_notes")
    lines.extend(["", "### Source notes", ""])
    if ambiguity_notes:
        lines.extend(f"- {value}" for value in ambiguity_notes)
    else:
        lines.append("- No annotation ambiguity was recorded.")

    table_text = record.get(_string(profile, "table_text_field"))
    if not isinstance(table_text, str):
        raise ValueError(f"{question_id} source table text is missing")
    blocks = extract_markdown_tables(table_text, min_pipe_count=min_pipe_count)
    referenced_tables = {table_index}
    referenced_tables.update(
        cast(int, cell.get("table_index", table_index)) for cell in valid_cells + near_misses
    )
    for referenced_table in sorted(referenced_tables):
        try:
            block = blocks[referenced_table]
        except IndexError as error:
            raise ValueError(f"{question_id} references a missing table") from error
        lines.extend(
            [
                "",
                f"### Complete relevant table T{referenced_table}",
                "",
                *block,
            ]
        )

    context = record.get(context_field)
    if not isinstance(context, str):
        raise ValueError(f"{question_id} source context is missing")
    lines.extend(
        [
            "",
            "<details>",
            "<summary>Open the complete extracted source context</summary>",
            "",
            f"<pre>{html.escape(context)}</pre>",
            "",
            "</details>",
            "",
            "**Reviewer note:**",
            "",
            "---",
            "",
        ]
    )
    return lines


def make_source_review_pack(config: Config) -> Path:
    """Validate the pilot and write its complete human source-review pack."""

    repository = Path(__file__).resolve().parents[1]
    validate_pilot_annotations(config, repository)
    pilot = _mapping(config.get("pilot"), "pilot")
    audit = _mapping(config.get("audit"), "audit")
    datasets = _mapping(config.get("datasets"), "datasets")
    primary = _mapping(datasets.get("primary"), "datasets.primary")
    field_names = _mapping(audit.get("field_names"), "audit.field_names")
    annotations_root = _load_yaml_mapping(
        _repository_path(repository, _string(pilot, "annotation_path"))
    )
    manifest = _load_yaml_mapping(_repository_path(repository, _string(primary, "manifest_path")))
    annotations = [
        _mapping(value, "pilot annotation")
        for value in _list(annotations_root.get("annotations"), "annotations")
    ]
    profiles = _source_profiles(manifest)
    requested_ids: dict[SourceKey, set[str]] = defaultdict(set)
    for annotation in annotations:
        source = _mapping(annotation.get("source"), "annotation source")
        requested_ids[(_string(source, "subset"), _string(source, "manifest_split"))].add(
            _string(annotation, "question_id")
        )
    records, _ = _load_requested_records(
        repository=repository,
        profiles=profiles,
        requested_ids=requested_ids,
        id_field=_string(field_names, "id"),
    )
    individually_reviewed, delegated_reviewed = _source_statuses(annotations_root)
    metadata = create_run_metadata(config, repository)
    lines = [
        "# Human source-review pack — exact-cell pilot",
        "",
        f"- Generated at: `{metadata.timestamp}`",
        f"- Git commit: `{metadata.git_commit}`",
        f"- Config hash: `{metadata.config_hash}`",
        f"- Questions: {len(annotations)}",
        "",
        "## How to review",
        "",
        "For every question, read the question, proposed correct cell, complete relevant table, "
        "source notes and—when needed—the expandable full context. Mark:",
        "",
        "- **INCLUDE** when a clearly identified printed cell answers the question without "
        "calculation;",
        "- **EXCLUDE** when the answer requires calculation or the source meaning is unclear; or",
        "- **UNSURE** when you need a second opinion.",
        "",
        "Reviewing only the proposed answer is not full-source inspection. The complete table "
        "and any context needed to resolve its meaning must be checked.",
        "",
        "## Review index",
        "",
    ]
    for index, annotation in enumerate(annotations, start=1):
        question_id = _string(annotation, "question_id")
        lines.append(f"{index}. [`{question_id}`](#question-{index})")
    lines.extend(["", "---", ""])

    context_field = _string(field_names, "context")
    min_pipe_count = _integer(audit, "markdown_min_pipe_count")
    for index, annotation in enumerate(annotations, start=1):
        question_id = _string(annotation, "question_id")
        source = _mapping(annotation.get("source"), f"{question_id}.source")
        source_key = (_string(source, "subset"), _string(source, "manifest_split"))
        record = records[(source_key[0], source_key[1], question_id)]
        lines.extend(
            _render_question(
                ordinal=index,
                annotation=annotation,
                record=record,
                profile=profiles[source_key],
                context_field=context_field,
                min_pipe_count=min_pipe_count,
                individually_reviewed=individually_reviewed,
                delegated_reviewed=delegated_reviewed,
            )
        )

    lines.extend(
        [
            "## Final reviewer declaration",
            "",
            "After inspecting all 30 source sections, report:",
            "",
            "- Included question IDs:",
            "- Excluded question IDs and reasons:",
            "- Unsure question IDs:",
            "- Reviewer name:",
            "- Review completion date:",
            "",
        ]
    )
    output_path = _repository_path(repository, _string(pilot, "review_pack_output_path"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--condition-config", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config, _ = load_config(
        cast(Path, args.base_config),
        cast(Path, args.condition_config),
    )
    output_path = make_source_review_pack(config)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

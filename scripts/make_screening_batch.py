"""Create a fixed prospective screening batch and its human review pack."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import yaml

from scripts.audit_t2_ragbench import (
    extract_markdown_tables,
    scalar_numeric,
)
from scripts.build_pilot_goldset import (
    SourceKey,
    _boolean,
    _integer,
    _list,
    _load_requested_records,
    _load_yaml_mapping,
    _mapping,
    _repository_path,
    _source_profiles,
    _string,
)
from src.config import Config, load_config
from src.results import create_run_metadata
from src.tables.normalize import numeric_values
from src.types import RunMetadata


def _stable_rank(candidate: dict[str, Any], seed: int) -> str:
    identity = ":".join(
        (
            str(seed),
            _string(candidate, "subset"),
            _string(candidate, "split"),
            _string(candidate, "id"),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def select_candidates(
    candidates: list[dict[str, Any]],
    *,
    excluded_ids: set[str],
    subset_targets: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    """Select a deterministic, stratified batch without human eligibility filtering."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for candidate in candidates:
        question_id = _string(candidate, "id")
        if question_id in seen_ids:
            raise ValueError(f"Duplicate automatic candidate ID: {question_id}")
        seen_ids.add(question_id)
        if question_id not in excluded_ids:
            grouped[_string(candidate, "subset")].append(candidate)

    selected: list[dict[str, Any]] = []
    for subset, target in subset_targets.items():
        available = sorted(grouped.get(subset, []), key=lambda item: _stable_rank(item, seed))
        if len(available) < target:
            raise ValueError(
                f"Not enough unseen candidates for {subset}: {len(available)} < {target}"
            )
        selected.extend(available[:target])
    return selected


def _load_audit_result(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        root = _mapping(json.load(handle), str(path))
    result = _mapping(root.get("result"), "audit result")
    return root | {"result": result}


def _prior_question_ids(path: Path) -> set[str]:
    root = _load_yaml_mapping(path)
    return {
        _string(_mapping(item, "prior annotation"), "question_id")
        for item in _list(root.get("annotations"), "prior annotations")
    }


def _subset_targets(screening: dict[str, Any]) -> dict[str, int]:
    configured = _mapping(screening.get("subset_targets"), "screening.subset_targets")
    targets: dict[str, int] = {}
    for subset, target in configured.items():
        if not isinstance(subset, str) or not subset:
            raise ValueError("Screening subset names must be non-empty strings")
        if not isinstance(target, int) or isinstance(target, bool) or target < 1:
            raise ValueError(f"Screening target for {subset} must be a positive integer")
        targets[subset] = target
    return targets


def _selection_records(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "question_id": _string(candidate, "id"),
            "subset": _string(candidate, "subset"),
            "manifest_split": _string(candidate, "split"),
            "context_id": _string(candidate, "context_id"),
            "decision": None,
            "reason_codes": [],
            "notes": None,
        }
        for candidate in selected
    ]


def _write_or_validate_decision_log(
    path: Path,
    *,
    screening: dict[str, Any],
    selected: list[dict[str, Any]],
    metadata: RunMetadata,
    audit_root: dict[str, Any],
    seed: int,
    excluded_prior_annotation_count: int,
) -> None:
    selected_records = _selection_records(selected)
    selected_ids = [record["question_id"] for record in selected_records]
    if path.exists():
        existing = _load_yaml_mapping(path)
        if _string(existing, "batch_id") != _string(screening, "batch_id"):
            raise ValueError("Existing screening log has a different batch ID")
        existing_selection = _mapping(existing.get("selection"), "existing selection")
        if _string(existing_selection, "method") != _string(screening, "selection_method"):
            raise ValueError("Existing screening log has a different selection method")
        if _integer(existing_selection, "seed") != seed:
            raise ValueError("Existing screening log has a different selection seed")
        existing_records = [
            _mapping(item, "existing screening decision")
            for item in _list(existing.get("decisions"), "existing screening decisions")
        ]
        existing_ids = [_string(record, "question_id") for record in existing_records]
        if existing_ids != selected_ids:
            raise ValueError(
                "Existing screening log has a different fixed selection; refusing to overwrite it"
            )
        return

    audit_metadata = _mapping(audit_root.get("run_metadata"), "audit run metadata")
    log = {
        "schema_version": 1,
        "batch_id": _string(screening, "batch_id"),
        "status": "awaiting_human_screening",
        "generation": {
            "timestamp": metadata.timestamp,
            "git_commit": metadata.git_commit,
            "config_hash": metadata.config_hash,
        },
        "selection": {
            "method": _string(screening, "selection_method"),
            "seed": seed,
            "excluded_prior_annotation_count": excluded_prior_annotation_count,
            "human_prefiltering_before_selection": False,
            "audit_git_commit": _string(audit_metadata, "git_commit"),
            "audit_config_hash": _string(audit_metadata, "config_hash"),
        },
        "review": {
            "reviewer": None,
            "started_at": None,
            "completed_at": None,
            "active_minutes": None,
            "instructions": (
                "Start a timer before item 1. Review every item in order and stop the timer "
                "after the final decision. Do not replace difficult candidates."
            ),
        },
        "allowed_decisions": _list(screening.get("decisions"), "screening decisions"),
        "allowed_exclusion_reason_codes": _list(
            screening.get("exclusion_reason_codes"), "screening exclusion reason codes"
        ),
        "decisions": selected_records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(log, handle, sort_keys=False, allow_unicode=True)


def _answer_match_lines(record: dict[str, Any], table_text: str, min_pipe_count: int) -> list[str]:
    answer_values = {
        value
        for value in (
            scalar_numeric(record.get("program_answer")),
            scalar_numeric(record.get("original_answer")),
        )
        if value is not None
    }
    matches: list[str] = []
    for table_index, block in enumerate(
        extract_markdown_tables(table_text, min_pipe_count=min_pipe_count)
    ):
        display_row = 0
        for line in block:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if cells and all(set(cell) <= {"-", ":", " "} for cell in cells):
                continue
            for column_index, cell in enumerate(cells):
                if any(value in answer_values for value in numeric_values(cell)):
                    matches.append(
                        f"- T{table_index}, displayed row {display_row}, raw column "
                        f"{column_index}: `{cell}`"
                    )
            display_row += 1
    return matches or ["- No automatic matching cell could be reconstructed."]


def _markdown_table_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _assistant_suggestions(decision_log_path: Path) -> dict[str, tuple[str, str]]:
    root = _load_yaml_mapping(decision_log_path)
    suggestions: dict[str, tuple[str, str]] = {}
    for item in _list(root.get("decisions"), "screening decisions"):
        decision = _mapping(item, "screening decision")
        question_id = _string(decision, "question_id")
        suggestion = decision.get("assistant_suggestion")
        reason = decision.get("assistant_reason")
        if suggestion is None and reason is None:
            continue
        if not isinstance(suggestion, str) or suggestion not in {"include", "exclude", "unsure"}:
            raise ValueError(f"Invalid assistant suggestion for {question_id}")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"Missing assistant reason for {question_id}")
        suggestions[question_id] = (suggestion, reason)
    return suggestions


def _render_pack(
    *,
    config: Config,
    screening: dict[str, Any],
    selected: list[dict[str, Any]],
    records: dict[tuple[str, str, str], dict[str, Any]],
    profiles: dict[SourceKey, dict[str, Any]],
    audit_root: dict[str, Any],
    output_path: Path,
    decision_log_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    audit = _mapping(config.get("audit"), "audit")
    fields = _mapping(audit.get("field_names"), "audit.field_names")
    min_pipe_count = _integer(audit, "markdown_min_pipe_count")
    metadata = create_run_metadata(config, repository)
    audit_metadata = _mapping(audit_root.get("run_metadata"), "audit run metadata")
    suggestions = _assistant_suggestions(decision_log_path)
    lines = [
        f"# Prospective exact-cell screening — {_string(screening, 'batch_id')}",
        "",
        f"- Batch ID: `{_string(screening, 'batch_id')}`",
        f"- Generated at: `{metadata.timestamp}`",
        f"- Git commit: `{metadata.git_commit}`",
        f"- Config hash: `{metadata.config_hash}`",
        f"- Source audit commit: `{_string(audit_metadata, 'git_commit')}`",
        f"- Decision log: `{decision_log_path.relative_to(repository)}`",
        f"- Candidates: {len(selected)}",
        "",
        "## Before you begin",
        "",
        "1. Start a stopwatch immediately before reviewing item 1.",
        "2. Review every item in order. Do not skip or replace difficult items.",
        "3. Mark **INCLUDE** only if one printed table cell answers the question directly, "
        "without calculation.",
        "4. Mark **EXCLUDE** if calculation is required, the answer is not a cell, or the "
        "correct cell cannot be identified clearly. Record the reason.",
        "5. Mark **UNSURE** when a second opinion is genuinely needed.",
        f"6. Stop the stopwatch after item {len(selected)} and report the total active minutes.",
        "",
        "The automatic match is only a clue. It may be wrong—for example, a row number can "
        "accidentally equal the dataset answer.",
        "",
        "My suggestions are advisory pre-labels, not gold labels. Please challenge them. "
        "Every disagreement will be retained in the decision log.",
        "",
        "## Compact critique table",
        "",
        "| # | Question ID | Question | Proposed answer | Codex suggestion | Why | Your decision |",
        "|---:|---|---|---:|---|---|---|",
    ]
    for ordinal, candidate in enumerate(selected, start=1):
        question_id = _string(candidate, "id")
        suggestion, reason = suggestions.get(question_id, ("not provided", "not provided"))
        lines.append(
            "| "
            + " | ".join(
                (
                    str(ordinal),
                    f"`{_markdown_table_cell(question_id)}`",
                    _markdown_table_cell(_string(candidate, "question")),
                    _markdown_table_cell(candidate.get("original_answer")),
                    suggestion.upper(),
                    _markdown_table_cell(reason),
                    "",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "---",
            "",
        ]
    )
    context_field = _string(fields, "context")
    for ordinal, candidate in enumerate(selected, start=1):
        subset = _string(candidate, "subset")
        split = _string(candidate, "split")
        question_id = _string(candidate, "id")
        profile = profiles[(subset, split)]
        record = records[(subset, split, question_id)]
        table_text = record.get(_string(profile, "table_text_field"))
        context = record.get(context_field)
        if not isinstance(table_text, str) or not isinstance(context, str):
            raise ValueError(f"Missing source text for {question_id}")
        page = record.get(_string(fields, "page_number"))
        page_text = "unavailable" if page is None else str(page)
        suggestion, suggestion_reason = suggestions.get(
            question_id, ("not provided", "No advisory suggestion was recorded.")
        )
        lines.extend(
            [
                f"## {ordinal}. `{question_id}`",
                "",
                "**Decision:** ☐ INCLUDE &nbsp;&nbsp; ☐ EXCLUDE &nbsp;&nbsp; ☐ UNSURE",
                "",
                "**Reason/notes:**",
                "",
                f"**Codex suggestion (advisory):** {suggestion.upper()}",
                "",
                f"**Codex reason:** {suggestion_reason}",
                "",
                f"**Dataset:** {subset} · split `{split}`",
                "",
                f"**Question:** {_string(candidate, 'question')}",
                "",
                f"**Dataset answers:** program=`{candidate.get('program_answer')}` · "
                f"original=`{candidate.get('original_answer')}`",
                "",
                f"**Source:** `{candidate.get('file_name')}` · page {page_text} · "
                f"context `{candidate.get('context_id')}`",
                "",
                "### Automatically matched cell location(s)",
                "",
                *_answer_match_lines(record, table_text, min_pipe_count),
                "",
                "### Complete source table(s)",
                "",
            ]
        )
        for block in extract_markdown_tables(table_text, min_pipe_count=min_pipe_count):
            lines.extend([*block, ""])
        lines.extend(
            [
                "<details>",
                "<summary>Open the complete extracted source context</summary>",
                "",
                f"<pre>{html.escape(context)}</pre>",
                "",
                "</details>",
                "",
                "---",
                "",
            ]
        )
    lines.extend(
        [
            "## Completion record",
            "",
            "- Reviewer:",
            "- Start time:",
            "- Finish time:",
            "- Total active minutes:",
            "- Included IDs:",
            "- Excluded IDs and reason codes:",
            "- Unsure IDs:",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def make_screening_batch(config: Config) -> tuple[Path, Path]:
    """Create or reproduce the configured prospective screening batch."""

    repository = Path(__file__).resolve().parents[1]
    screening = _mapping(config.get("screening"), "screening")
    if not _boolean(screening, "enabled"):
        raise ValueError("screening.enabled must be true")
    project = _mapping(config.get("project"), "project")
    seed = _integer(project, "seed")
    targets = _subset_targets(screening)
    if sum(targets.values()) != _integer(screening, "sample_size"):
        raise ValueError("screening.sample_size must equal the sum of subset_targets")

    audit_path = _repository_path(repository, _string(screening, "audit_result_path"))
    audit_root = _load_audit_result(audit_path)
    result = _mapping(audit_root.get("result"), "audit result")
    candidates = [
        _mapping(item, "automatic screening candidate")
        for item in _list(result.get("pilot_candidates"), "automatic screening candidates")
    ]
    prior_path = _repository_path(repository, _string(screening, "prior_annotation_path"))
    prior_ids = _prior_question_ids(prior_path)
    selected = select_candidates(
        candidates,
        excluded_ids=prior_ids,
        subset_targets=targets,
        seed=seed,
    )

    datasets = _mapping(config.get("datasets"), "datasets")
    primary = _mapping(datasets.get("primary"), "datasets.primary")
    result_dataset = _mapping(result.get("dataset"), "audit result dataset")
    if _string(result_dataset, "revision") != _string(primary, "revision"):
        raise ValueError("Screening audit and configured primary dataset revisions differ")
    manifest = _load_yaml_mapping(_repository_path(repository, _string(primary, "manifest_path")))
    profiles = _source_profiles(manifest)
    requested_ids: dict[SourceKey, set[str]] = defaultdict(set)
    for candidate in selected:
        requested_ids[(_string(candidate, "subset"), _string(candidate, "split"))].add(
            _string(candidate, "id")
        )
    audit = _mapping(config.get("audit"), "audit")
    fields = _mapping(audit.get("field_names"), "audit.field_names")
    records, _ = _load_requested_records(
        repository=repository,
        profiles=profiles,
        requested_ids=requested_ids,
        id_field=_string(fields, "id"),
    )

    metadata = create_run_metadata(config, repository)
    decision_log_path = _repository_path(repository, _string(screening, "decision_log_path"))
    _write_or_validate_decision_log(
        decision_log_path,
        screening=screening,
        selected=selected,
        metadata=metadata,
        audit_root=audit_root,
        seed=seed,
        excluded_prior_annotation_count=len(prior_ids),
    )
    output_path = _repository_path(repository, _string(screening, "review_pack_output_path"))
    _render_pack(
        config=config,
        screening=screening,
        selected=selected,
        records=records,
        profiles=profiles,
        audit_root=audit_root,
        output_path=output_path,
        decision_log_path=decision_log_path,
    )
    return output_path, decision_log_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--condition-config", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config, _ = load_config(cast(Path, args.base_config), cast(Path, args.condition_config))
    output_path, decision_log_path = make_screening_batch(config)
    print(output_path)
    print(decision_log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

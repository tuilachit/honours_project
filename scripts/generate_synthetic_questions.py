"""Generate deterministic direct-cell questions from pinned real financial tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import yaml

from scripts.audit_t2_ragbench import (
    SEPARATOR_CELL_PATTERN,
    YEAR_PATTERN,
    extract_markdown_tables,
    scalar_numeric,
    sha256_file,
)
from scripts.build_pilot_goldset import (
    _boolean,
    _integer,
    _list,
    _load_yaml_mapping,
    _mapping,
    _repository_path,
    _string,
    stable_cell_id,
)
from src.config import Config, load_config
from src.results import write_result_json
from src.types import RunMetadata

SPACE_PATTERN = re.compile(r"\s+")
TRAILING_FOOTNOTE_PATTERN = re.compile(r"\s+\([a-z]\)\s*$", re.IGNORECASE)
MARKDOWN_PATTERN = re.compile(r"[*_`]+")


@dataclass(frozen=True, slots=True)
class ParsedBlock:
    rows: tuple[tuple[str, ...], ...]
    raw_nonseparator_rows: tuple[int, ...]
    markdown: str


def _clean_text(value: str) -> str:
    cleaned = MARKDOWN_PATTERN.sub("", value)
    cleaned = TRAILING_FOOTNOTE_PATTERN.sub("", cleaned)
    return SPACE_PATTERN.sub(" ", cleaned).strip(" :;|")


def _display_entity(value: str) -> str:
    text = SPACE_PATTERN.sub(" ", value.replace("-", " ")).strip()
    if "-" not in value and text != text.lower():
        return text
    acronyms = {
        "ag": "AG",
        "bt": "BT",
        "llc": "LLC",
        "lng": "LNG",
        "lp": "LP",
        "nv": "N.V.",
        "plc": "PLC",
        "sa": "S.A.",
        "se": "SE",
        "te": "TE",
    }
    suffixes = {"corp": "Corp.", "inc": "Inc.", "ltd": "Ltd."}
    return " ".join(
        acronyms.get(token.lower(), suffixes.get(token.lower(), token.title()))
        for token in text.split()
    )


def _replace_company_mention(question: str, raw_entity: str, display_entity: str) -> str:
    return re.sub(re.escape(raw_entity), lambda _: display_entity, question, flags=re.IGNORECASE)


def _stable_digest(*parts: object) -> str:
    text = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _candidate_review_hash(question: dict[str, Any]) -> str:
    """Bind a human review decision to the exact generated candidate."""

    source = _mapping(question.get("source"), "candidate review source")
    payload = {
        "question_id": _string(question, "question_id"),
        "question": _string(question, "question"),
        "answer": question.get("answer"),
        "source": {
            key: source.get(key)
            for key in (
                "dataset",
                "subset",
                "manifest_split",
                "source_record_id",
                "context_id",
                "table_index",
                "table_sha256",
            )
        },
        "target_cell": _mapping(question.get("target_cell"), "candidate target cell"),
        "natural_hard_negatives": _list(
            question.get("natural_hard_negatives"), "candidate hard negatives"
        ),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_block(block: list[str], *, drop_leading_column: bool) -> ParsedBlock | None:
    rows: list[tuple[str, ...]] = []
    raw_rows: list[int] = []
    nonseparator_index = 0
    for line in block:
        cells = tuple(cell.strip() for cell in line.strip("|").split("|"))
        nonempty = [cell for cell in cells if cell]
        if nonempty and all(SEPARATOR_CELL_PATTERN.fullmatch(cell) for cell in nonempty):
            continue
        if drop_leading_column:
            if len(cells) < 2:
                return None
            cells = cells[1:]
        rows.append(cells)
        raw_rows.append(nonseparator_index)
        nonseparator_index += 1
    if not rows or len({len(row) for row in rows}) != 1:
        return None
    return ParsedBlock(tuple(rows), tuple(raw_rows), "\n".join(block))


def _header_row_count(rows: tuple[tuple[str, ...], ...]) -> int:
    count = 1
    while count < len(rows) - 1:
        row = rows[count]
        if row[0].strip() or not any(YEAR_PATTERN.search(cell) for cell in row[1:]):
            break
        count += 1
    return count


def _header_path(rows: tuple[tuple[str, ...], ...], header_count: int, column: int) -> str:
    return " / ".join(_header_path_parts(rows, header_count, column))


def _header_path_parts(
    rows: tuple[tuple[str, ...], ...], header_count: int, column: int
) -> tuple[str, ...]:
    parts: list[str] = []
    for row in rows[:header_count]:
        part = _clean_text(row[column])
        if part and (not parts or parts[-1] != part):
            parts.append(part)
    return tuple(parts)


def _valid_period(period: str, *, require_explicit_year: bool) -> bool:
    years = set(YEAR_PATTERN.findall(period))
    if require_explicit_year and len(years) != 1:
        return False
    return bool(period) and len(period) <= 100


def _concept_tokens(concept: str, minimum_characters: int) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z]+", concept.lower())
        if len(token) >= minimum_characters
    }


def _valid_concept(concept: str, generation: dict[str, Any]) -> bool:
    if not (
        _integer(generation, "minimum_concept_characters")
        <= len(concept)
        <= _integer(generation, "maximum_concept_characters")
    ):
        return False
    if not _boolean(generation, "allow_digits_in_concept") and any(
        character.isdigit() for character in concept
    ):
        return False
    normalized = concept.lower().rstrip(".")
    disallowed = {
        str(value).lower()
        for value in _list(generation.get("disallowed_exact_concepts"), "disallowed concepts")
    }
    if normalized in disallowed:
        return False
    words = re.findall(r"[a-z]+", normalized)
    if len(words) >= _integer(generation, "minimum_concept_word_count"):
        return True
    allowed_single = {
        str(value).lower()
        for value in _list(
            generation.get("allowed_single_token_concepts"), "allowed single-token concepts"
        )
    }
    return len(words) == 1 and words[0] in allowed_single


def _is_year_value(raw_value: str, number: Decimal) -> bool:
    stripped = raw_value.strip().replace(",", "")
    return stripped.isdigit() and Decimal(1900) <= number <= Decimal(2100)


def _source_exclusions(repository: Path, configured_paths: list[Any]) -> set[str]:
    excluded: set[str] = set()
    for configured in configured_paths:
        if not isinstance(configured, str) or not configured:
            raise ValueError("synthetic exclusion paths must be non-empty strings")
        root = _load_yaml_mapping(_repository_path(repository, configured))
        annotations = root.get("annotations")
        if isinstance(annotations, list):
            for item in annotations:
                annotation = _mapping(item, "excluded pilot annotation")
                source = _mapping(annotation.get("source"), "excluded annotation source")
                excluded.add(_string(source, "context_id"))
        decisions = root.get("decisions")
        if isinstance(decisions, list):
            for item in decisions:
                decision = _mapping(item, "excluded screening decision")
                excluded.add(_string(decision, "context_id"))
    return excluded


def _candidate_cells(
    *,
    block: ParsedBlock,
    header_count: int,
    generation: dict[str, Any],
) -> list[tuple[int, int, Decimal, str, str]]:
    rows = block.rows
    periods = {
        column: _header_path(rows, header_count, column) for column in range(1, len(rows[0]))
    }
    periods = {
        column: period
        for column, period in periods.items()
        if _valid_period(
            period,
            require_explicit_year=_boolean(generation, "require_explicit_year_header"),
        )
    }
    candidates: list[tuple[int, int, Decimal, str, str]] = []
    for row_index in range(header_count, len(rows)):
        concept = _clean_text(rows[row_index][0])
        if not _valid_concept(concept, generation):
            continue
        for column, period in periods.items():
            raw_value = rows[row_index][column]
            number = scalar_numeric(raw_value)
            if number is None or _is_year_value(raw_value, number):
                continue
            candidates.append((row_index, column, number, concept, period))
    return candidates


def _quality_candidate(
    *,
    repository_revision: str,
    seed: int,
    subset: str,
    split: str,
    record: dict[str, Any],
    block: ParsedBlock,
    table_index: int,
    drop_leading_column: bool,
    generation: dict[str, Any],
) -> dict[str, Any] | None:
    rows = block.rows
    header_count = _header_row_count(rows)
    cells = _candidate_cells(block=block, header_count=header_count, generation=generation)
    if not cells:
        return None

    source_question = _string(record, "question")
    source_question_lower = source_question.lower()
    exclusion_phrases = [
        str(value).lower()
        for value in _list(
            generation.get("source_question_exclusion_phrases"),
            "source question exclusion phrases",
        )
    ]
    if any(phrase in source_question_lower for phrase in exclusion_phrases):
        return None
    answer_values = {
        value
        for value in (
            scalar_numeric(record.get("program_answer")),
            scalar_numeric(record.get("original_answer")),
        )
        if value is not None
    }
    source_question_years = set(YEAR_PATTERN.findall(source_question))

    all_table_values = [
        number
        for row in rows[header_count:]
        for raw_value in row[1:]
        if (number := scalar_numeric(raw_value)) is not None
    ]
    by_value = Counter(all_table_values)
    eligible: list[dict[str, Any]] = []
    context_id = _string(record, "context_id")
    for row_index, column, value, concept, period in cells:
        if _boolean(generation, "require_source_answer_match") and value not in answer_values:
            continue
        period_years = set(YEAR_PATTERN.findall(period))
        if _boolean(
            generation, "require_source_question_year_match"
        ) and not period_years.intersection(source_question_years):
            continue
        report_years = set(YEAR_PATTERN.findall(str(record.get("report_year", ""))))
        if _boolean(
            generation,
            "require_source_question_years_within_target_or_report_year",
        ) and not source_question_years.issubset(period_years | report_years):
            continue
        concept_tokens = _concept_tokens(
            concept,
            _integer(generation, "minimum_concept_overlap_token_characters"),
        )
        modifier_evidence = f"{concept} {period} {rows[row_index][column]}".lower()
        configured_modifiers = _mapping(
            generation.get("source_modifier_evidence"),
            "source modifier evidence",
        )
        modifier_mismatch = False
        for raw_modifier, raw_evidence_terms in configured_modifiers.items():
            if not isinstance(raw_modifier, str) or not raw_modifier:
                raise ValueError("Source modifiers must be non-empty strings")
            evidence_terms = [
                str(value).lower()
                for value in _list(
                    raw_evidence_terms,
                    f"source modifier evidence for {raw_modifier}",
                )
            ]
            if raw_modifier.lower() in source_question_lower and not any(
                term in modifier_evidence for term in evidence_terms
            ):
                modifier_mismatch = True
                break
        if modifier_mismatch:
            continue
        entity_tokens = _concept_tokens(
            _display_entity(_string(record, "company_name")),
            _integer(generation, "minimum_concept_overlap_token_characters"),
        )
        if concept_tokens and concept_tokens.issubset(entity_tokens):
            continue
        overlap_count = sum(token in source_question_lower for token in concept_tokens)
        overlap_ratio = overlap_count / len(concept_tokens) if concept_tokens else 0.0
        configured_overlap = generation.get("minimum_concept_token_overlap_ratio")
        if not isinstance(configured_overlap, int | float) or isinstance(configured_overlap, bool):
            raise ValueError("minimum concept token overlap ratio must be numeric")
        if _boolean(
            generation, "require_source_question_concept_overlap"
        ) and overlap_ratio < float(configured_overlap):
            continue
        if (
            _boolean(generation, "require_unique_target_value_within_table")
            and by_value[value] != 1
        ):
            continue
        wrong_periods = [
            item
            for item in cells
            if item[0] == row_index and item[1] != column and item[2] != value
        ]
        wrong_concepts = [
            item
            for item in cells
            if item[1] == column and item[0] != row_index and item[2] != value
        ]
        if _boolean(generation, "require_wrong_period_negative") and not wrong_periods:
            continue
        if _boolean(generation, "require_wrong_concept_negative") and not wrong_concepts:
            continue
        wrong_period = min(
            wrong_periods,
            key=lambda item: _stable_digest(
                seed, context_id, table_index, row_index, column, "p", item
            ),
        )
        wrong_concept = min(
            wrong_concepts,
            key=lambda item: _stable_digest(
                seed, context_id, table_index, row_index, column, "c", item
            ),
        )
        eligible.append(
            {
                "row_index": row_index,
                "column_index": column,
                "number": value,
                "concept": concept,
                "period": period,
                "wrong_period": wrong_period,
                "wrong_concept": wrong_concept,
            }
        )
    if not eligible:
        return None
    chosen = min(
        eligible,
        key=lambda item: _stable_digest(
            seed, context_id, table_index, item["row_index"], item["column_index"]
        ),
    )
    row_index = cast(int, chosen["row_index"])
    column = cast(int, chosen["column_index"])
    concept = cast(str, chosen["concept"])
    period = cast(str, chosen["period"])
    period_year = YEAR_PATTERN.findall(period)[0]
    raw_entity = _string(record, "company_name")
    entity = _display_entity(raw_entity)
    templates = [
        str(value) for value in _list(generation.get("question_templates"), "question templates")
    ]
    if not templates:
        raise ValueError("At least one question template is required")
    template_digest = _stable_digest(
        seed, subset, split, context_id, table_index, row_index, column
    )
    template = templates[int(template_digest[:8], 16) % len(templates)]
    natural_source_question = _replace_company_mention(source_question, raw_entity, entity)
    source_question_lower = natural_source_question[0].lower() + natural_source_question[1:]
    question = template.format(
        entity=entity,
        concept=concept,
        period=period_year,
        source_question=natural_source_question,
        source_question_lower=source_question_lower,
    )
    cell_id = stable_cell_id(
        dataset_revision=repository_revision,
        subset=subset,
        manifest_split=split,
        context_id=context_id,
        table_index=table_index,
        row_index=row_index,
        column_index=column,
    )

    def negative(kind: str, item: tuple[int, int, Decimal, str, str]) -> dict[str, Any]:
        negative_row, negative_column, _, negative_concept, negative_period = item
        return {
            "kind": kind,
            "cell_id": stable_cell_id(
                dataset_revision=repository_revision,
                subset=subset,
                manifest_split=split,
                context_id=context_id,
                table_index=table_index,
                row_index=negative_row,
                column_index=negative_column,
            ),
            "row_index": negative_row,
            "column_index": negative_column,
            "raw_markdown_column_index": negative_column + int(drop_leading_column),
            "concept": negative_concept,
            "period": negative_period,
            "raw_column_header_path": list(_header_path_parts(rows, header_count, negative_column)),
            "raw_value": rows[negative_row][negative_column],
        }

    question_id = (
        "syn_"
        + _stable_digest(
            repository_revision, subset, split, context_id, table_index, row_index, column
        )[:20]
    )
    return {
        "question_id": question_id,
        "question": question,
        "answer": rows[row_index][column],
        "assistant_suggestion": _string(generation, "assistant_suggestion"),
        "assistant_reason": (
            "Approve: real source table, explicit single-year header, meaningful row label, "
            "unique target value, and both wrong-period and wrong-concept natural negatives."
        ),
        "source": {
            "dataset": "T2-RAGBench",
            "subset": subset,
            "manifest_split": split,
            "source_record_id": _string(record, "id"),
            "context_id": context_id,
            "file_name": _string(record, "file_name"),
            "page_number": record.get("page_number"),
            "company_name": _string(record, "company_name"),
            "report_year": record.get("report_year"),
            "source_original_question": source_question,
            "table_index": table_index,
            "table_sha256": hashlib.sha256(block.markdown.encode("utf-8")).hexdigest(),
        },
        "target_cell": {
            "cell_id": cell_id,
            "row_index": row_index,
            "column_index": column,
            "raw_nonseparator_row_index": block.raw_nonseparator_rows[row_index],
            "raw_markdown_column_index": column + int(drop_leading_column),
            "raw_row_header_path": [concept],
            "raw_column_header_path": list(_header_path_parts(rows, header_count, column)),
            "raw_value": rows[row_index][column],
        },
        "natural_hard_negatives": [
            negative(
                "wrong_period", cast(tuple[int, int, Decimal, str, str], chosen["wrong_period"])
            ),
            negative(
                "wrong_concept", cast(tuple[int, int, Decimal, str, str], chosen["wrong_concept"])
            ),
        ],
        "quality_checks": {
            "source_integrity_verified": True,
            "one_question_per_table": True,
            "explicit_single_year_header": True,
            "meaningful_concept_header": True,
            "unique_target_value_within_table": True,
            "wrong_period_negative_present": True,
            "wrong_concept_negative_present": True,
            "source_answer_matches_target": True,
            "source_question_year_matches": True,
            "source_question_years_within_target_or_report_year": True,
            "source_question_concept_overlaps": True,
            "source_question_modifiers_have_cell_evidence": True,
            "configured_calculation_phrase_absent": True,
            "human_verified": False,
        },
        "source_table_markdown": block.markdown,
        "selection_rank": _stable_digest(
            seed, subset, split, context_id, table_index, row_index, column
        ),
    }


def _source_profiles(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = _mapping(manifest.get("dataset"), "dataset manifest")
    return [_mapping(item, "dataset source") for item in _list(dataset.get("files"), "files")]


def _scan_candidates(
    *,
    repository: Path,
    manifest: dict[str, Any],
    generation: dict[str, Any],
    config: Config,
    excluded_contexts: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    project = _mapping(config.get("project"), "project")
    seed = _integer(project, "seed")
    datasets = _mapping(config.get("datasets"), "datasets")
    primary = _mapping(datasets.get("primary"), "datasets.primary")
    revision = _string(primary, "revision")
    pilot = _mapping(config.get("pilot"), "pilot")
    transforms = _mapping(pilot.get("coordinate_transforms"), "coordinate transforms")
    best_by_table: dict[tuple[str, str, str], dict[str, Any]] = {}
    integrity: list[dict[str, Any]] = []
    for profile in _source_profiles(manifest):
        subset = _string(profile, "subset")
        split = _string(profile, "split")
        path = _repository_path(repository, _string(profile, "path"))
        if path.stat().st_size != _integer(profile, "expected_bytes"):
            raise ValueError(f"Byte-count mismatch for {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != _string(profile, "sha256"):
            raise ValueError(f"SHA-256 mismatch for {path}")
        integrity.append(
            {
                "subset": subset,
                "split": split,
                "path": str(path.relative_to(repository)),
                "sha256": actual_sha256,
                "verified": True,
            }
        )
        transform = _mapping(transforms.get(subset), f"coordinate transform for {subset}")
        drop_leading = _boolean(transform, "drop_leading_synthetic_column")
        table_text_field = _string(profile, "table_text_field")
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = _mapping(json.loads(line), f"{path}:{line_number}")
                context_id = record.get("context_id")
                if not isinstance(context_id, str) or context_id in excluded_contexts:
                    continue
                if not isinstance(record.get("question"), str) or not record["question"].strip():
                    continue
                table_text = record.get(table_text_field)
                if not isinstance(table_text, str) or not table_text:
                    continue
                for table_index, raw_block in enumerate(
                    extract_markdown_tables(
                        table_text,
                        min_pipe_count=_integer(
                            _mapping(config.get("audit"), "audit"), "markdown_min_pipe_count"
                        ),
                    )
                ):
                    table_hash = hashlib.sha256("\n".join(raw_block).encode("utf-8")).hexdigest()
                    table_key = (subset, context_id, table_hash)
                    block = _parse_block(raw_block, drop_leading_column=drop_leading)
                    if block is None:
                        continue
                    row_count = len(block.rows)
                    column_count = len(block.rows[0])
                    if not (
                        _integer(generation, "minimum_table_rows")
                        <= row_count
                        <= _integer(generation, "maximum_table_rows")
                        and _integer(generation, "minimum_table_columns")
                        <= column_count
                        <= _integer(generation, "maximum_table_columns")
                    ):
                        continue
                    candidate = _quality_candidate(
                        repository_revision=revision,
                        seed=seed,
                        subset=subset,
                        split=split,
                        record=record,
                        block=block,
                        table_index=table_index,
                        drop_leading_column=drop_leading,
                        generation=generation,
                    )
                    if candidate is not None:
                        existing = best_by_table.get(table_key)
                        if existing is None or _string(candidate, "selection_rank") < _string(
                            existing, "selection_rank"
                        ):
                            best_by_table[table_key] = candidate
    return list(best_by_table.values()), integrity


def _subset_targets(generation: dict[str, Any]) -> dict[str, int]:
    configured = _mapping(generation.get("subset_targets"), "synthetic subset targets")
    targets: dict[str, int] = {}
    for subset, value in configured.items():
        if not isinstance(subset, str) or not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("Synthetic subset targets must map strings to integers")
        if value < 0:
            raise ValueError("Synthetic subset targets must be non-negative")
        targets[subset] = value
    if sum(targets.values()) != _integer(generation, "sample_size"):
        raise ValueError("Synthetic sample size must equal the sum of subset targets")
    return targets


def select_synthetic_questions(
    candidates: list[dict[str, Any]],
    *,
    targets: dict[str, int],
    maximum_per_company: int,
) -> list[dict[str, Any]]:
    """Select deterministic subset-balanced questions with a company diversity cap."""

    if maximum_per_company < 1:
        raise ValueError("maximum_per_company must be at least one")

    selected: list[dict[str, Any]] = []
    company_counts: Counter[tuple[str, str]] = Counter()
    seen_question_texts: set[str] = set()
    for subset, target in targets.items():
        if target == 0:
            continue
        available = sorted(
            (
                candidate
                for candidate in candidates
                if _string(_mapping(candidate.get("source"), "candidate source"), "subset")
                == subset
            ),
            key=lambda candidate: _string(candidate, "selection_rank"),
        )
        subset_selected: list[dict[str, Any]] = []
        for candidate in available:
            source = _mapping(candidate.get("source"), "candidate source")
            company = _string(source, "company_name")
            normalized_question = SPACE_PATTERN.sub(
                " ", _string(candidate, "question").lower()
            ).strip()
            if normalized_question in seen_question_texts:
                continue
            company_key = (subset, company)
            if company_counts[company_key] >= maximum_per_company:
                continue
            subset_selected.append(candidate)
            company_counts[company_key] += 1
            seen_question_texts.add(normalized_question)
            if len(subset_selected) == target:
                break
        if len(subset_selected) != target:
            raise ValueError(
                f"Only {len(subset_selected)} high-quality candidates available for {subset}; "
                f"target is {target}"
            )
        selected.extend(subset_selected)
    return selected


def _load_validated_review_log(
    path: Path,
    *,
    questions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    root = _load_yaml_mapping(path)
    reviews = [_mapping(item, "synthetic review") for item in _list(root.get("reviews"), "reviews")]
    expected_ids = [_string(question, "question_id") for question in questions]
    if [_string(review, "question_id") for review in reviews] != expected_ids:
        raise ValueError("Existing synthetic review log has a different deterministic selection")
    expected_hashes = [_candidate_review_hash(question) for question in questions]
    actual_hashes = [review.get("candidate_sha256") for review in reviews]
    if actual_hashes != expected_hashes:
        raise ValueError("Existing synthetic review log was created for different question content")
    return root


def _apply_review_decisions(
    questions: list[dict[str, Any]],
    *,
    review_root: dict[str, Any] | None,
    allowed_decisions: set[str],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if review_root is None:
        return {"approved": 0, "edited": 0, "rejected": 0, "unsure": 0, "pending": len(questions)}

    reviews = [
        _mapping(item, "synthetic review") for item in _list(review_root.get("reviews"), "reviews")
    ]
    for question, review in zip(questions, reviews, strict=True):
        raw_decision = review.get("human_decision")
        if raw_decision is None:
            decision: str | None = None
            counts["pending"] += 1
        elif not isinstance(raw_decision, str) or raw_decision not in allowed_decisions:
            raise ValueError(f"Invalid synthetic human decision: {raw_decision!r}")
        else:
            decision = raw_decision
            counts[decision] += 1
        question["human_review"] = {
            "decision": decision,
            "edited_question": review.get("human_edited_question"),
            "notes": review.get("human_notes"),
        }
        quality_checks = _mapping(question.get("quality_checks"), "question quality checks")
        quality_checks["human_verified"] = decision == "approve"
    return {
        "approved": counts["approve"],
        "edited": counts["edit"],
        "rejected": counts["reject"],
        "unsure": counts["unsure"],
        "pending": counts["pending"],
    }


def _write_or_validate_review_log(
    path: Path,
    *,
    generation: dict[str, Any],
    questions: list[dict[str, Any]],
    metadata: RunMetadata,
) -> None:
    if _load_validated_review_log(path, questions=questions) is not None:
        return
    suggestion = _string(generation, "assistant_suggestion")
    root = {
        "schema_version": 1,
        "dataset_id": _string(generation, "dataset_id"),
        "status": "awaiting_human_verification",
        "generation": {
            "timestamp": metadata.timestamp,
            "git_commit": metadata.git_commit,
            "config_hash": metadata.config_hash,
        },
        "review_protocol": {
            "assistant_suggestions_are_advisory": True,
            "human_decision_is_gold": True,
            "blind_independent_review_required_later": True,
            "allowed_human_decisions": _list(generation.get("human_decisions"), "human decisions"),
        },
        "reviews": [
            {
                "question_id": _string(question, "question_id"),
                "candidate_sha256": _candidate_review_hash(question),
                "assistant_suggestion": suggestion,
                "assistant_reason": _string(question, "assistant_reason"),
                "human_decision": None,
                "human_edited_question": None,
                "human_notes": None,
            }
            for question in questions
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(root, handle, sort_keys=False, allow_unicode=True, width=120)


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _write_review_pack(
    path: Path,
    *,
    generation: dict[str, Any],
    questions: list[dict[str, Any]],
    metadata: RunMetadata,
    review_log_path: Path,
    repository: Path,
) -> None:
    human_decisions_recorded = sum(
        "human_review" in question
        and _mapping(question.get("human_review"), "human review").get("decision") is not None
        for question in questions
    )
    lines = [
        f"# Human review pack — {len(questions)} deterministic synthetic questions",
        "",
        f"- Dataset ID: `{_string(generation, 'dataset_id')}`",
        f"- Generated at: `{metadata.timestamp}`",
        f"- Git commit: `{metadata.git_commit}`",
        f"- Config hash: `{metadata.config_hash}`",
        f"- Questions: {len(questions)}",
        f"- Human decisions recorded: {human_decisions_recorded} / {len(questions)}",
        f"- Decision log: `{review_log_path.relative_to(repository)}`",
        "",
        (
            "The financial tables and values are real. Only the questions and controlled "
            "target/negative mappings are generated. Codex suggestions are advisory; mark "
            "APPROVE, EDIT, REJECT or UNSURE after checking the source table."
        ),
        "",
        "## All questions",
        "",
        (
            "| # | ID | Source | Generated question | Target | Wrong period | "
            "Wrong concept | Codex | Your decision |"
        ),
        "|---:|---|---|---|---:|---:|---:|---|---|",
    ]
    for ordinal, question in enumerate(questions, start=1):
        source = _mapping(question.get("source"), "source")
        target = _mapping(question.get("target_cell"), "target cell")
        negatives = [
            _mapping(item, "hard negative")
            for item in _list(question.get("natural_hard_negatives"), "hard negatives")
        ]
        human_review = question.get("human_review")
        human_decision = (
            _mapping(human_review, "human review").get("decision")
            if isinstance(human_review, dict)
            else None
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    str(ordinal),
                    f"[`{_string(question, 'question_id')}`](#q-{ordinal})",
                    _markdown_cell(f"{source.get('subset')} · {source.get('company_name')}"),
                    _markdown_cell(_string(question, "question")),
                    _markdown_cell(target.get("raw_value")),
                    _markdown_cell(negatives[0].get("raw_value")),
                    _markdown_cell(negatives[1].get("raw_value")),
                    _string(question, "assistant_suggestion").upper(),
                    str(human_decision).upper() if human_decision is not None else "",
                )
            )
            + " |"
        )
    lines.extend(["", "## Source details", ""])
    for ordinal, question in enumerate(questions, start=1):
        source = _mapping(question.get("source"), "source")
        target = _mapping(question.get("target_cell"), "target")
        negatives = [
            _mapping(item, "negative")
            for item in _list(question.get("natural_hard_negatives"), "negatives")
        ]
        human_review = question.get("human_review")
        human_decision = (
            _mapping(human_review, "human review").get("decision")
            if isinstance(human_review, dict)
            else None
        )
        target_row = (
            f"| Target | {_markdown_cell(target.get('raw_row_header_path'))} | "
            f"{_markdown_cell(target.get('raw_column_header_path'))} | "
            f"{_markdown_cell(target.get('raw_value'))} | "
            f"r{target.get('row_index')},c{target.get('column_index')} |"
        )
        wrong_period_row = (
            f"| Wrong period | {_markdown_cell(negatives[0].get('concept'))} | "
            f"{_markdown_cell(negatives[0].get('period'))} | "
            f"{_markdown_cell(negatives[0].get('raw_value'))} | "
            f"r{negatives[0].get('row_index')},c{negatives[0].get('column_index')} |"
        )
        wrong_concept_row = (
            f"| Wrong concept | {_markdown_cell(negatives[1].get('concept'))} | "
            f"{_markdown_cell(negatives[1].get('period'))} | "
            f"{_markdown_cell(negatives[1].get('raw_value'))} | "
            f"r{negatives[1].get('row_index')},c{negatives[1].get('column_index')} |"
        )
        source_line = (
            f"**Source:** `{source.get('file_name')}` · page {source.get('page_number')} · "
            f"context `{source.get('context_id')}` · table {source.get('table_index')}"
        )
        lines.extend(
            [
                f'<a id="q-{ordinal}"></a>',
                "",
                f"### {ordinal}. `{_string(question, 'question_id')}`",
                "",
                f"**Generated question:** {_string(question, 'question')}",
                "",
                f"**Codex suggestion:** {_string(question, 'assistant_suggestion').upper()}",
                "",
                f"**Why:** {_string(question, 'assistant_reason')}",
                "",
                "| Role | Concept | Period | Value | Cell address |",
                "|---|---|---|---:|---|",
                target_row,
                wrong_period_row,
                wrong_concept_row,
                "",
                source_line,
                "",
                "**Your decision:** "
                + " &nbsp; ".join(
                    f"{'☒' if human_decision == decision else '☐'} {decision.upper()}"
                    for decision in ("approve", "edit", "reject", "unsure")
                ),
                "",
                "**Your criticism/edited question:**",
                "",
                "<details>",
                "<summary>Open the complete source table</summary>",
                "",
                _string(question, "source_table_markdown"),
                "",
                "</details>",
                "",
                "---",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_synthetic_questions(config: Config) -> tuple[Path, Path, Path]:
    repository = Path(__file__).resolve().parents[1]
    generation = _mapping(config.get("synthetic_generation"), "synthetic_generation")
    if not _boolean(generation, "enabled"):
        raise ValueError("synthetic_generation.enabled must be true")
    if _string(generation, "table_hash_algorithm").lower() != "sha256":
        raise ValueError("Only sha256 is supported for synthetic table hashes")
    if _string(generation, "cell_id_hash_algorithm").lower() != "sha256":
        raise ValueError("Only sha256 is supported for synthetic cell IDs")
    datasets = _mapping(config.get("datasets"), "datasets")
    primary = _mapping(datasets.get("primary"), "datasets.primary")
    manifest_path = _repository_path(repository, _string(primary, "manifest_path"))
    manifest = _load_yaml_mapping(manifest_path)
    excluded_contexts = _source_exclusions(
        repository,
        _list(generation.get("excluded_annotation_paths"), "excluded annotation paths"),
    )
    candidates, integrity = _scan_candidates(
        repository=repository,
        manifest=manifest,
        generation=generation,
        config=config,
        excluded_contexts=excluded_contexts,
    )
    targets = _subset_targets(generation)
    questions = select_synthetic_questions(
        candidates,
        targets=targets,
        maximum_per_company=_integer(generation, "maximum_examples_per_company"),
    )
    for question in questions:
        question.pop("selection_rank", None)
    review_log_path = _repository_path(repository, _string(generation, "decision_log_path"))
    review_root = _load_validated_review_log(review_log_path, questions=questions)
    allowed_decisions = {
        str(value) for value in _list(generation.get("human_decisions"), "human decisions")
    }
    review_summary = _apply_review_decisions(
        questions,
        review_root=review_root,
        allowed_decisions=allowed_decisions,
    )
    review_complete = review_summary["pending"] == 0
    result = {
        "dataset_id": _string(generation, "dataset_id"),
        "status": (
            "first_pass_human_review_complete"
            if review_complete
            else "synthetic_candidates_awaiting_human_verification"
        ),
        "generation_method": _string(generation, "selection_method"),
        "source_dataset_revision": _string(primary, "revision"),
        "excluded_prior_context_count": len(excluded_contexts),
        "automatic_candidate_count": len(candidates),
        "question_count": len(questions),
        "subset_counts": dict(
            Counter(_mapping(q.get("source"), "source")["subset"] for q in questions)
        ),
        "human_review_summary": review_summary,
        "quality_contract": {
            "real_tables_and_values_only": True,
            "questions_are_synthetic": True,
            "human_verification_required": True,
            "first_pass_human_review_complete": review_complete,
            "assistant_suggestions_are_not_gold": True,
        },
        "source_integrity": integrity,
        "questions": questions,
    }
    output_path = _repository_path(repository, _string(generation, "output_path"))
    metadata = write_result_json(
        output_path,
        result,
        resolved_config=config,
        repository=repository,
        dataset_manifest_paths={"primary": manifest_path},
    )
    _write_or_validate_review_log(
        review_log_path,
        generation=generation,
        questions=questions,
        metadata=metadata,
    )
    review_pack_path = _repository_path(repository, _string(generation, "review_pack_output_path"))
    _write_review_pack(
        review_pack_path,
        generation=generation,
        questions=questions,
        metadata=metadata,
        review_log_path=review_log_path,
        repository=repository,
    )
    return output_path, review_pack_path, review_log_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--condition-config", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config, _ = load_config(cast(Path, args.base_config), cast(Path, args.condition_config))
    for path in generate_synthetic_questions(config):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

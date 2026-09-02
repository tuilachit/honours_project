"""Materialise approved synthetic tables as FinancialFacts and verify lossless recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from scripts.build_pilot_goldset import (
    _boolean,
    _integer,
    _list,
    _load_yaml_mapping,
    _mapping,
    _repository_path,
    _string,
)
from src.config import Config, load_config
from src.facts import (
    build_financial_facts,
    deserialize_financial_fact,
    serialize_financial_fact,
)
from src.hashing import sha256_file
from src.results import write_result_json
from src.tables import normalize_table, parse_source_table
from src.types import FinancialFact, SourceTable


def _page_number(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise ValueError(f"Unsupported page number: {value!r}")


def _table_id(source: dict[str, Any]) -> str:
    payload = {
        "context_id": _string(source, "context_id"),
        "manifest_split": _string(source, "manifest_split"),
        "subset": _string(source, "subset"),
        "table_index": _integer(source, "table_index"),
        "table_sha256": _string(source, "table_sha256"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _source_table(
    question: dict[str, Any],
    *,
    config: Config,
    dataset_revision: str,
) -> SourceTable:
    source = _mapping(question.get("source"), "question source")
    pilot = _mapping(config.get("pilot"), "pilot config")
    transforms = _mapping(pilot.get("coordinate_transforms"), "coordinate transforms")
    subset = _string(source, "subset")
    transform = _mapping(transforms.get(subset), f"coordinate transform for {subset}")
    table_id = _table_id(source)
    return SourceTable(
        dataset=_string(source, "dataset"),
        raw_text=_string(question, "source_table_markdown"),
        table_id=table_id,
        document_id=_string(source, "file_name"),
        context_id=_string(source, "context_id"),
        page_number=_page_number(source.get("page_number")),
        source_uri=_string(source, "file_name"),
        source_format="markdown",
        metadata={
            "dataset_revision": dataset_revision,
            "subset": subset,
            "manifest_split": _string(source, "manifest_split"),
            "context_id": _string(source, "context_id"),
            "table_index": _integer(source, "table_index"),
            "table_sha256": _string(source, "table_sha256"),
            "entity": _string(source, "company_name"),
            "drop_leading_synthetic_column": _boolean(transform, "drop_leading_synthetic_column"),
            "cell_id_hash_algorithm": _string(
                _mapping(config.get("synthetic_generation"), "synthetic generation"),
                "cell_id_hash_algorithm",
            ),
        },
    )


def _validate_label(
    label: Mapping[str, object],
    *,
    fact_by_cell_id: Mapping[str, FinancialFact],
    role: str,
) -> dict[str, object]:
    cell_id_value = label.get("cell_id")
    if not isinstance(cell_id_value, str) or not cell_id_value:
        raise ValueError(f"{role} label is missing a cell ID")
    fact = fact_by_cell_id.get(cell_id_value)
    if fact is None:
        raise ValueError(f"{role} cell was not materialised: {cell_id_value}")
    expected_raw_value = label.get("raw_value")
    if fact.raw_value != expected_raw_value:
        raise ValueError(
            f"{role} raw-value mismatch for {cell_id_value}: "
            f"{fact.raw_value!r} != {expected_raw_value!r}"
        )
    raw_row_headers = label.get("raw_row_header_path")
    if raw_row_headers is None:
        concept = label.get("concept")
        raw_row_headers = [concept] if isinstance(concept, str) and concept else []
    if not isinstance(raw_row_headers, list) or not all(
        isinstance(value, str) for value in raw_row_headers
    ):
        raise ValueError(f"{role} row-header path must be a string list")
    raw_column_headers = label.get("raw_column_header_path")
    if not isinstance(raw_column_headers, list) or not all(
        isinstance(value, str) for value in raw_column_headers
    ):
        raise ValueError(f"{role} column-header path must be a string list")
    if fact.source_address.row_header_path != tuple(raw_row_headers):
        raise ValueError(f"{role} row-header mismatch for {cell_id_value}")
    if fact.source_address.column_header_path != tuple(raw_column_headers):
        raise ValueError(f"{role} column-header mismatch for {cell_id_value}")
    return {
        "role": role,
        "cell_id": cell_id_value,
        "fact_id": fact.fact_id,
        "raw_value": fact.raw_value,
        "recovered": True,
    }


def materialize_fact_roundtrip(config: Config, repository: Path) -> dict[str, Any]:
    """Build table-wide facts first, then validate approved labels against them."""

    settings = _mapping(config.get("fact_roundtrip"), "fact_roundtrip config")
    if not _boolean(settings, "enabled"):
        raise ValueError("fact_roundtrip.enabled must be true")
    input_path = _repository_path(repository, _string(settings, "input_path"))
    review_path = _repository_path(repository, _string(settings, "review_log_path"))
    input_envelope = _mapping(json.loads(input_path.read_text(encoding="utf-8")), "input result")
    input_result = _mapping(input_envelope.get("result"), "input result payload")
    if _string(input_result, "status") != _string(settings, "required_input_status"):
        raise ValueError("Synthetic input has not completed the required first-pass review")
    questions = [
        _mapping(item, "synthetic question")
        for item in _list(input_result.get("questions"), "synthetic questions")
    ]
    if len(questions) != _integer(settings, "expected_question_count"):
        raise ValueError("Synthetic question count does not match the configured expectation")

    review_root = _load_yaml_mapping(review_path)
    if _string(review_root, "status") != _string(settings, "required_review_status"):
        raise ValueError("Synthetic review log has not completed first-pass review")
    reviews = [
        _mapping(item, "synthetic review")
        for item in _list(review_root.get("reviews"), "synthetic reviews")
    ]
    if len(reviews) != len(questions):
        raise ValueError("Synthetic review count does not match the question count")
    review_by_id = {_string(review, "question_id"): review for review in reviews}
    if len(review_by_id) != len(reviews):
        raise ValueError("Synthetic review log contains duplicate question IDs")
    required_decision = _string(settings, "required_human_decision")
    for question in questions:
        question_id = _string(question, "question_id")
        review = review_by_id.get(question_id)
        if review is None or review.get("human_decision") != required_decision:
            raise ValueError(f"Question is not approved in the review log: {question_id}")

    datasets = _mapping(config.get("datasets"), "datasets config")
    primary = _mapping(datasets.get("primary"), "primary dataset config")
    dataset_revision = _string(primary, "revision")
    if _string(input_result, "source_dataset_revision") != dataset_revision:
        raise ValueError("Synthetic input dataset revision does not match the active config")
    source_tables: dict[str, SourceTable] = {}
    for question in questions:
        source_table = _source_table(
            question,
            config=config,
            dataset_revision=dataset_revision,
        )
        existing = source_tables.get(source_table.table_id)
        if existing is not None and existing != source_table:
            raise ValueError(f"Conflicting source table identity: {source_table.table_id}")
        source_tables[source_table.table_id] = source_table
    if len(source_tables) != _integer(settings, "expected_source_table_count"):
        raise ValueError("Source-table count does not match the configured expectation")

    facts: list[FinancialFact] = []
    parsed_table_summaries: list[dict[str, object]] = []
    for source_table in source_tables.values():
        parsed = normalize_table(parse_source_table(source_table, config), config)
        if parsed.metadata.get("gold_used_for_parsing") is not False:
            raise ValueError("Parser did not attest label-independent materialisation")
        table_facts = build_financial_facts(parsed, config)
        facts.extend(table_facts)
        parsed_table_summaries.append(
            {
                "table_id": source_table.table_id,
                "context_id": source_table.context_id,
                "parsed_numeric_fact_count": len(table_facts),
                "gold_used_for_parsing": False,
            }
        )
    if len({fact.fact_id for fact in facts}) != len(facts):
        raise ValueError("FinancialFact IDs are not unique")
    if len({fact.source_address.cell_id for fact in facts}) != len(facts):
        raise ValueError("Materialised cell IDs are not unique")
    if len(facts) != _integer(settings, "expected_materialized_fact_count"):
        raise ValueError("Materialised fact count does not match the configured expectation")

    serialized_facts = [serialize_financial_fact(fact, config) for fact in facts]
    restored_facts = [deserialize_financial_fact(payload, config) for payload in serialized_facts]
    if restored_facts != facts:
        raise ValueError("FinancialFact serialization round trip changed fact content")
    fact_by_cell_id = {fact.source_address.cell_id: fact for fact in restored_facts}

    target_checks: list[dict[str, object]] = []
    hard_negative_checks: list[dict[str, object]] = []
    required_negatives = _integer(settings, "required_hard_negative_count_per_question")
    for question in questions:
        target_checks.append(
            _validate_label(
                _mapping(question.get("target_cell"), "target cell"),
                fact_by_cell_id=fact_by_cell_id,
                role="target",
            )
        )
        negatives = [
            _mapping(item, "hard negative")
            for item in _list(question.get("natural_hard_negatives"), "hard negatives")
        ]
        if len(negatives) != required_negatives:
            raise ValueError("Question has an unexpected hard-negative count")
        hard_negative_checks.extend(
            _validate_label(
                negative,
                fact_by_cell_id=fact_by_cell_id,
                role=_string(negative, "kind"),
            )
            for negative in negatives
        )
    if len(target_checks) != _integer(settings, "expected_target_count"):
        raise ValueError("Recovered target count does not match the configured expectation")
    if len(hard_negative_checks) != _integer(settings, "expected_hard_negative_count"):
        raise ValueError("Recovered hard-negative count does not match the configured expectation")

    return {
        "status": "financial_fact_roundtrip_passed",
        "source_dataset_revision": dataset_revision,
        "input_artifact_sha256": sha256_file(input_path),
        "review_log_sha256": sha256_file(review_path),
        "human_review_status": _string(review_root, "status"),
        "question_count": len(questions),
        "source_table_count": len(source_tables),
        "materialized_fact_count": len(facts),
        "serialized_fact_count": len(serialized_facts),
        "target_recovery_count": len(target_checks),
        "hard_negative_recovery_count": len(hard_negative_checks),
        "gold_used_for_parsing": False,
        "fact_id_unique": True,
        "cell_id_unique": True,
        "serialization_roundtrip_exact": True,
        "subset_counts": dict(
            Counter(
                _string(_mapping(question.get("source"), "question source"), "subset")
                for question in questions
            )
        ),
        "parsed_tables": parsed_table_summaries,
        "target_checks": target_checks,
        "hard_negative_checks": hard_negative_checks,
        "facts": serialized_facts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--condition-config", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config, _ = load_config(cast(Path, args.base_config), cast(Path, args.condition_config))
    repository = Path(__file__).resolve().parents[1]
    result = materialize_fact_roundtrip(config, repository)
    settings = _mapping(config.get("fact_roundtrip"), "fact_roundtrip config")
    output_path = _repository_path(repository, _string(settings, "output_path"))
    datasets = _mapping(config.get("datasets"), "datasets config")
    primary = _mapping(datasets.get("primary"), "primary dataset config")
    manifest_path = _repository_path(repository, _string(primary, "manifest_path"))
    write_result_json(
        output_path,
        result,
        resolved_config=config,
        repository=repository,
        dataset_manifest_paths={"primary": manifest_path},
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

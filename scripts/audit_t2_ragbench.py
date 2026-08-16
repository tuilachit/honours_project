"""Audit a pinned T2-RAGBench release for exact-cell study feasibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, cast

from src.config import Config, load_config
from src.results import write_result_json

YEAR_PATTERN = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
NUMBER_PATTERN = re.compile(
    r"(?<![\w.])(?P<open>\()?\s*(?:[$£€])?\s*"
    r"(?P<number>[+-]?\d[\d,]*(?:\.\d+)?)\s*%?\s*(?P<close>\))?"
)
SEPARATOR_CELL_PATTERN = re.compile(r"^:?-{2,}:?$")


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


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_markdown_tables(text: str, *, min_pipe_count: int) -> list[list[str]]:
    """Return consecutive Markdown-style table lines as separate blocks."""

    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        is_table_line = stripped.startswith("|") and stripped.count("|") >= min_pipe_count
        if is_table_line:
            current.append(stripped)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def table_cells(blocks: list[list[str]]) -> list[str]:
    """Extract non-separator cells from Markdown table blocks."""

    cells: list[str] = []
    for block in blocks:
        for line in block:
            row = [cell.strip() for cell in line.strip("|").split("|")]
            nonempty = [cell for cell in row if cell]
            if nonempty and all(SEPARATOR_CELL_PATTERN.fullmatch(cell) for cell in nonempty):
                continue
            cells.extend(nonempty)
    return cells


def numeric_values(value: object) -> list[Decimal]:
    """Extract displayed numeric values while preserving accounting negatives."""

    if value is None:
        return []
    text = str(value).replace("−", "-").replace("–", "-")
    parsed: list[Decimal] = []
    for match in NUMBER_PATTERN.finditer(text):
        number_text = match.group("number").replace(",", "")
        try:
            number = Decimal(number_text)
        except InvalidOperation:
            continue
        if match.group("open") and match.group("close") and number > 0:
            number = -number
        parsed.append(number)
    return parsed


def scalar_numeric(value: object) -> Decimal | None:
    """Return one unambiguous numeric answer, otherwise None."""

    values = numeric_values(value)
    unique = list(dict.fromkeys(values))
    if len(unique) != 1:
        return None
    return unique[0]


@dataclass
class SourceAudit:
    """Mutable counters and identifiers for one source file."""

    subset: str
    split: str
    path: str
    rows: int = 0
    markdown_rows: int = 0
    scalar_program_answers: int = 0
    program_answer_cell_match: int = 0
    any_answer_cell_match: int = 0
    unique_answer_cell_match: int = 0
    multiple_answer_cell_match: int = 0
    answer_in_context_not_cell: int = 0
    natural_confuser_candidates: int = 0
    requested_year_in_multiyear_table: int = 0
    table_blocks: int = 0
    numeric_cells: int = 0
    context_ids: set[str] = field(default_factory=set)
    file_names: set[str] = field(default_factory=set)
    companies: set[str] = field(default_factory=set)
    field_presence: Counter[str] = field(default_factory=Counter)
    pilot_candidates: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Return JSON-serialisable audit counts."""

        return {
            "subset": self.subset,
            "split": self.split,
            "path": self.path,
            "rows": self.rows,
            "distinct_contexts": len(self.context_ids),
            "distinct_files": len(self.file_names),
            "distinct_companies": len(self.companies),
            "rows_with_markdown_tables": self.markdown_rows,
            "table_blocks": self.table_blocks,
            "numeric_cells": self.numeric_cells,
            "scalar_program_answers": self.scalar_program_answers,
            "program_answer_cell_match": self.program_answer_cell_match,
            "any_answer_cell_match": self.any_answer_cell_match,
            "unique_answer_cell_match": self.unique_answer_cell_match,
            "multiple_answer_cell_match": self.multiple_answer_cell_match,
            "answer_in_context_not_cell": self.answer_in_context_not_cell,
            "natural_confuser_candidates": self.natural_confuser_candidates,
            "requested_year_in_multiyear_table": self.requested_year_in_multiyear_table,
            "field_presence": dict(sorted(self.field_presence.items())),
        }


def _row_text(row: dict[str, Any], field_name: str) -> str:
    value = row.get(field_name)
    return value if isinstance(value, str) else ""


def audit_source(
    repository: Path,
    source: dict[str, Any],
    audit_config: dict[str, Any],
) -> tuple[SourceAudit, dict[str, Any]]:
    """Validate and audit one pinned JSONL source."""

    subset = _string(source, "subset")
    split = _string(source, "split")
    relative_path = _string(source, "path")
    path = repository / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    expected_bytes = _integer(source, "expected_bytes")
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Byte-count mismatch for {relative_path}: {actual_bytes} != {expected_bytes}"
        )

    expected_sha256 = _string(source, "sha256")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {relative_path}")

    required_fields = [
        str(item) for item in _list(audit_config.get("common_required_fields"), "required fields")
    ]
    fields = _mapping(audit_config.get("field_names"), "field names")
    id_field = _string(fields, "id")
    context_id_field = _string(fields, "context_id")
    question_field = _string(fields, "question")
    program_field = _string(fields, "program_answer")
    original_field = _string(fields, "original_answer")
    context_field = _string(fields, "context")
    file_field = _string(fields, "file_name")
    company_field = _string(fields, "company_name")
    table_text_field = _string(source, "table_text_field")
    min_pipe_count = _integer(audit_config, "markdown_min_pipe_count")
    pilot_limit = _integer(audit_config, "pilot_candidates_per_source")

    result = SourceAudit(subset=subset, split=split, path=relative_path)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected object at {relative_path}:{line_number}")
            row = cast(dict[str, Any], parsed)
            result.rows += 1
            for field_name in required_fields:
                if field_name in row and row[field_name] not in (None, ""):
                    result.field_presence[field_name] += 1

            context_id = row.get(context_id_field)
            file_name = row.get(file_field)
            company = row.get(company_field)
            if isinstance(context_id, str) and context_id:
                result.context_ids.add(context_id)
            if isinstance(file_name, str) and file_name:
                result.file_names.add(file_name)
            if isinstance(company, str) and company:
                result.companies.add(company)

            table_text = _row_text(row, table_text_field)
            context_text = _row_text(row, context_field)
            blocks = extract_markdown_tables(table_text, min_pipe_count=min_pipe_count)
            if not blocks:
                continue
            result.markdown_rows += 1
            result.table_blocks += len(blocks)
            cells = table_cells(blocks)
            cell_numbers = [number for cell in cells for number in numeric_values(cell)]
            result.numeric_cells += len(cell_numbers)

            program_answer = scalar_numeric(row.get(program_field))
            original_answer = scalar_numeric(row.get(original_field))
            if program_answer is not None:
                result.scalar_program_answers += 1
            if program_answer is not None and program_answer in cell_numbers:
                result.program_answer_cell_match += 1

            answer_values = {
                value for value in (program_answer, original_answer) if value is not None
            }
            cell_match_count = sum(number in answer_values for number in cell_numbers)
            if cell_match_count == 0:
                context_numbers = numeric_values(context_text)
                if answer_values and any(value in context_numbers for value in answer_values):
                    result.answer_in_context_not_cell += 1
                continue

            result.any_answer_cell_match += 1
            if cell_match_count == 1:
                result.unique_answer_cell_match += 1
            else:
                result.multiple_answer_cell_match += 1

            has_confuser = len(cell_numbers) > cell_match_count
            if has_confuser:
                result.natural_confuser_candidates += 1

            question = _row_text(row, question_field)
            question_years = set(YEAR_PATTERN.findall(question))
            table_years = sorted(set(YEAR_PATTERN.findall(table_text)))
            is_multiyear = len(table_years) >= 2 and bool(question_years.intersection(table_years))
            if is_multiyear:
                result.requested_year_in_multiyear_table += 1

            if (
                len(result.pilot_candidates) < pilot_limit
                and cell_match_count == 1
                and has_confuser
                and is_multiyear
            ):
                result.pilot_candidates.append(
                    {
                        "subset": subset,
                        "split": split,
                        "id": row.get(id_field),
                        "context_id": context_id,
                        "question": question,
                        "program_answer": row.get(program_field),
                        "original_answer": row.get(original_field),
                        "file_name": file_name,
                        "table_years": table_years,
                        "cell_match_count": cell_match_count,
                    }
                )

    expected_rows = _integer(source, "expected_rows")
    if result.rows != expected_rows:
        raise ValueError(
            f"Row-count mismatch for {relative_path}: {result.rows} != {expected_rows}"
        )

    integrity = {
        "path": relative_path,
        "rows": result.rows,
        "bytes": actual_bytes,
        "sha256": actual_sha256,
        "verified": True,
    }
    return result, integrity


def run_audit(config: Config, repository: Path) -> dict[str, Any]:
    """Run the full configured dataset audit."""

    dataset = _mapping(config.get("dataset"), "dataset")
    audit_config = _mapping(config.get("audit"), "audit")
    source_configs = [
        _mapping(item, "dataset source")
        for item in _list(dataset.get("files"), "dataset files")
    ]

    audits: list[SourceAudit] = []
    integrity: list[dict[str, Any]] = []
    for source in source_configs:
        audited, verified = audit_source(repository, source, audit_config)
        audits.append(audited)
        integrity.append(verified)

    all_contexts = set().union(*(audit.context_ids for audit in audits))
    all_files = set().union(*(audit.file_names for audit in audits))
    all_companies = set().union(*(audit.companies for audit in audits))
    total_rows = sum(audit.rows for audit in audits)
    expected_total_rows = _integer(dataset, "expected_total_rows")
    expected_contexts = _integer(dataset, "expected_distinct_contexts")
    if total_rows != expected_total_rows:
        raise ValueError(f"Total row mismatch: {total_rows} != {expected_total_rows}")
    if len(all_contexts) != expected_contexts:
        raise ValueError(f"Distinct-context mismatch: {len(all_contexts)} != {expected_contexts}")

    aggregate_keys = (
        "markdown_rows",
        "scalar_program_answers",
        "program_answer_cell_match",
        "any_answer_cell_match",
        "unique_answer_cell_match",
        "multiple_answer_cell_match",
        "answer_in_context_not_cell",
        "natural_confuser_candidates",
        "requested_year_in_multiyear_table",
        "table_blocks",
        "numeric_cells",
    )
    aggregate = {key: sum(int(getattr(audit, key)) for audit in audits) for key in aggregate_keys}
    aggregate.update(
        {
            "rows": total_rows,
            "distinct_contexts": len(all_contexts),
            "distinct_files": len(all_files),
            "distinct_companies": len(all_companies),
        }
    )

    return {
        "dataset": {
            "name": _string(dataset, "name"),
            "revision": _string(dataset, "revision"),
            "release": _string(dataset, "release"),
            "license": _string(dataset, "license"),
            "download_scope": _string(dataset, "download_scope"),
        },
        "integrity": integrity,
        "aggregate": aggregate,
        "sources": [audit.summary() for audit in audits],
        "pilot_candidates": [candidate for audit in audits for candidate in audit.pilot_candidates],
        "interpretation_limits": [
            "An exact numeric cell match is an automatic eligibility candidate, not a gold label.",
            (
                "Calculated answers can coincidentally equal a displayed cell and require human "
                "screening."
            ),
            (
                "The metadata download excludes source PDFs and images; visual verification is a "
                "later step."
            ),
            (
                "TAT-DQA stores one or more Markdown tables inside context rather than a separate "
                "table field."
            ),
        ],
    }


def parse_args() -> argparse.Namespace:
    """Parse required configuration paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--dataset-config", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Load configuration, run the audit, and write a stamped result."""

    args = parse_args()
    base_config = cast(Path, args.base_config)
    dataset_config = cast(Path, args.dataset_config)
    config, _ = load_config(base_config, dataset_config)
    repository = Path(__file__).resolve().parents[1]
    result = run_audit(config, repository)
    audit_config = _mapping(config.get("audit"), "audit")
    output_path = repository / _string(audit_config, "output_path")
    metadata = write_result_json(
        output_path,
        result,
        resolved_config=config,
        repository=repository,
        dataset_manifest_paths={"primary": dataset_config},
    )
    summary = {
        "output": str(output_path.relative_to(repository)),
        "git_commit": metadata.git_commit,
        "config_hash": metadata.config_hash,
        "aggregate": result["aggregate"],
        "pilot_candidates": len(cast(list[Any], result["pilot_candidates"])),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

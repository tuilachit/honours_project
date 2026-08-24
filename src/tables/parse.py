"""Schema-flexible Markdown table parsing with stable cell provenance."""

from __future__ import annotations

import re
from collections.abc import Mapping

from src.config import Config
from src.identity import stable_cell_id
from src.tables.normalize import parse_numeric_value
from src.types import CellAddress, DatasetCorpus, ParsedCell, ParsedTable, SourceTable

YEAR_PATTERN = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
SEPARATOR_CELL_PATTERN = re.compile(r"^:?-{2,}:?$")
MARKDOWN_PATTERN = re.compile(r"[*_`]+")
SPACE_PATTERN = re.compile(r"\s+")
TRAILING_FOOTNOTE_PATTERN = re.compile(r"\s+\([a-z]\)\s*$", re.IGNORECASE)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _integer(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def clean_table_header(value: str) -> str:
    """Apply the shared lossless header cleanup used by generation and parsing."""

    cleaned = MARKDOWN_PATTERN.sub("", value)
    cleaned = TRAILING_FOOTNOTE_PATTERN.sub("", cleaned)
    return SPACE_PATTERN.sub(" ", cleaned).strip(" :;|")


def _rows(source: SourceTable, *, minimum_pipe_count: int) -> tuple[tuple[str, ...], ...]:
    rows: list[tuple[str, ...]] = []
    drop_leading = source.metadata.get("drop_leading_synthetic_column", False)
    if not isinstance(drop_leading, bool):
        raise ValueError("drop_leading_synthetic_column metadata must be boolean")
    for line in source.raw_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.count("|") < minimum_pipe_count:
            continue
        cells = tuple(cell.strip() for cell in stripped.strip("|").split("|"))
        nonempty = [cell for cell in cells if cell]
        if nonempty and all(SEPARATOR_CELL_PATTERN.fullmatch(cell) for cell in nonempty):
            continue
        if drop_leading:
            if len(cells) < 2:
                raise ValueError("Cannot remove a synthetic column from a one-cell row")
            cells = cells[1:]
        rows.append(cells)
    if not rows:
        raise ValueError(f"No Markdown rows found in source table {source.table_id}")
    if len({len(row) for row in rows}) != 1:
        raise ValueError(f"Inconsistent Markdown row widths in source table {source.table_id}")
    return tuple(rows)


def _header_row_count(rows: tuple[tuple[str, ...], ...]) -> int:
    count = 1
    while count < len(rows) - 1:
        row = rows[count]
        if row[0].strip() or not any(YEAR_PATTERN.search(cell) for cell in row[1:]):
            break
        count += 1
    return count


def _column_header_path(
    rows: tuple[tuple[str, ...], ...], header_count: int, column_index: int
) -> tuple[str, ...]:
    parts: list[str] = []
    for row in rows[:header_count]:
        part = clean_table_header(row[column_index])
        if part and (not parts or parts[-1] != part):
            parts.append(part)
    return tuple(parts)


def parse_source_table(source: SourceTable, config: Config) -> ParsedTable:
    """Parse all configured data cells before consulting any gold annotation."""

    parsing = _mapping(config.get("table_parsing"), "table_parsing config")
    if _string(parsing, "source_format") != "markdown":
        raise ValueError("Only configured Markdown table parsing is currently supported")
    numeric_only = parsing.get("numeric_cells_only")
    if not isinstance(numeric_only, bool):
        raise ValueError("numeric_cells_only must be boolean")
    metadata = _mapping(source.metadata, "source table metadata")
    if _string(metadata, "cell_id_hash_algorithm").lower() != "sha256":
        raise ValueError("Only sha256 cell IDs are supported")
    rows = _rows(source, minimum_pipe_count=_integer(parsing, "markdown_min_pipe_count"))
    header_count = _header_row_count(rows)
    cells: list[ParsedCell] = []
    for row_index in range(header_count, len(rows)):
        row_header = clean_table_header(rows[row_index][0])
        for column_index in range(1, len(rows[row_index])):
            raw_value = rows[row_index][column_index]
            normalized_value = parse_numeric_value(raw_value)
            if numeric_only and normalized_value is None:
                continue
            address = CellAddress(
                cell_id=stable_cell_id(
                    dataset_revision=_string(metadata, "dataset_revision"),
                    subset=_string(metadata, "subset"),
                    manifest_split=_string(metadata, "manifest_split"),
                    context_id=_string(metadata, "context_id"),
                    table_index=_integer(metadata, "table_index"),
                    row_index=row_index,
                    column_index=column_index,
                ),
                dataset=source.dataset,
                document_id=source.document_id,
                context_id=source.context_id,
                page_number=source.page_number,
                table_id=source.table_id,
                row_index=row_index,
                column_index=column_index,
                row_header_path=(row_header,) if row_header else (),
                column_header_path=_column_header_path(rows, header_count, column_index),
                metadata={
                    "subset": _string(metadata, "subset"),
                    "manifest_split": _string(metadata, "manifest_split"),
                    "table_index": _integer(metadata, "table_index"),
                },
            )
            cells.append(
                ParsedCell(
                    address=address,
                    raw_value=raw_value,
                    normalized_value=normalized_value,
                )
            )
    return ParsedTable(
        source=source,
        cells=tuple(cells),
        table_id=source.table_id,
        row_count=len(rows),
        column_count=len(rows[0]),
        metadata={"header_row_count": header_count, "gold_used_for_parsing": False},
    )


def parse_corpus_tables(corpus: DatasetCorpus, config: Config) -> tuple[ParsedTable, ...]:
    """Parse each deduplicated source table once for an entire corpus snapshot."""

    return tuple(parse_source_table(source, config) for source in corpus.source_tables)

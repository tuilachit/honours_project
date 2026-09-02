"""Lossless parsed-cell normalisation for financial tables."""

from __future__ import annotations

import re
from dataclasses import replace
from decimal import Decimal, InvalidOperation

from src.config import Config
from src.types import ParsedCell, ParsedTable

NUMBER_PATTERN = re.compile(
    r"(?<![\w.])(?P<open>\()?\s*(?:[$£€])?\s*"
    r"(?P<number>[+-]?\d[\d,]*(?:\.\d+)?)\s*%?\s*(?P<close>\))?"
)


def numeric_values(value: object) -> list[Decimal]:
    """Extract displayed numbers while preserving accounting negatives."""

    if value is None:
        return []
    text = str(value).replace("−", "-").replace("–", "-")
    values: list[Decimal] = []
    for match in NUMBER_PATTERN.finditer(text):
        try:
            number = Decimal(match.group("number").replace(",", ""))
        except InvalidOperation:
            continue
        if match.group("open") and match.group("close") and number > 0:
            number = -number
        values.append(number)
    return values


def parse_numeric_value(value: object) -> Decimal | None:
    """Parse exactly one unambiguous displayed number."""

    unique_values = list(dict.fromkeys(numeric_values(value)))
    return unique_values[0] if len(unique_values) == 1 else None


def normalize_cell(cell: ParsedCell, config: Config) -> ParsedCell:
    """Attach an exact decimal without changing any raw source field."""

    normalisation = config.get("normalisation")
    if not isinstance(normalisation, dict):
        raise ValueError("normalisation config must be a mapping")
    if normalisation.get("preserve_raw_values") is not True:
        raise ValueError("FinancialFact normalisation requires preserve_raw_values=true")
    return replace(cell, normalized_value=parse_numeric_value(cell.raw_value))


def normalize_table(table: ParsedTable, config: Config) -> ParsedTable:
    """Normalise every parsed cell while retaining the original table source."""

    return replace(table, cells=tuple(normalize_cell(cell, config) for cell in table.cells))

"""Parsed table and cell normalisation interfaces."""

from src.config import Config
from src.types import ParsedCell, ParsedTable


def normalize_cell(cell: ParsedCell, config: Config) -> ParsedCell:
    """Normalise a cell value without discarding raw text or source headers."""

    raise NotImplementedError


def normalize_table(table: ParsedTable, config: Config) -> ParsedTable:
    """Normalise one parsed table according to the configured dataset schema."""

    raise NotImplementedError

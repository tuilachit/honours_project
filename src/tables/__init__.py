"""Table parsing and normalisation interfaces."""

from src.tables.normalize import normalize_cell, normalize_table
from src.tables.parse import parse_corpus_tables, parse_source_table

__all__ = [
    "normalize_cell",
    "normalize_table",
    "parse_corpus_tables",
    "parse_source_table",
]

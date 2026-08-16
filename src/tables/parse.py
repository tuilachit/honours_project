"""Schema-flexible source-table parsing interfaces."""

from src.config import Config
from src.types import DatasetCorpus, ParsedTable, SourceTable


def parse_source_table(source: SourceTable, config: Config) -> ParsedTable:
    """Parse one raw source table while preserving structural provenance."""

    raise NotImplementedError


def parse_corpus_tables(corpus: DatasetCorpus, config: Config) -> tuple[ParsedTable, ...]:
    """Parse each deduplicated source table once for an entire corpus snapshot."""

    raise NotImplementedError

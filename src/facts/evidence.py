"""Bridges from source tables and financial facts to retrievable evidence units."""

from collections.abc import Iterable

from src.config import Config
from src.types import EvidenceUnit, FinancialFact, SourceTable


def build_flattened_evidence(
    source_tables: Iterable[SourceTable],
    config: Config,
) -> Iterable[EvidenceUnit]:
    """Build B0 chunks with stable IDs and every contained source fact ID."""

    raise NotImplementedError


def build_fact_evidence(
    facts: Iterable[FinancialFact],
    config: Config,
) -> Iterable[EvidenceUnit]:
    """Build one canonical FINANCIAL_FACT evidence unit per source fact."""

    raise NotImplementedError

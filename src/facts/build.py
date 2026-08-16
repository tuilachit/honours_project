"""Canonical financial-fact construction interface."""

from src.config import Config
from src.types import FinancialFact, ParsedTable


def build_financial_facts(table: ParsedTable, config: Config) -> tuple[FinancialFact, ...]:
    """Build addressable financial facts from one normalised parsed table."""

    raise NotImplementedError

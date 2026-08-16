"""Financial-fact serialisation interfaces."""

from collections.abc import Mapping

from src.config import Config
from src.types import FinancialFact


def serialize_financial_fact(fact: FinancialFact, config: Config) -> Mapping[str, object]:
    """Convert a financial fact to a configured persistence representation."""

    raise NotImplementedError


def deserialize_financial_fact(
    payload: Mapping[str, object],
    config: Config,
) -> FinancialFact:
    """Restore a financial fact from a configured persistence representation."""

    raise NotImplementedError

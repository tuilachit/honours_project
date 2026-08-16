"""Schema-flexible structured indexing and retrieval interfaces."""

from collections.abc import Iterable, Mapping, Sequence

from src.config import Config
from src.types import (
    FactCandidate,
    FinancialFact,
    IndexArtifact,
    QueryContext,
    Question,
    RankedFact,
)


def build_structured_index(
    facts: Iterable[FinancialFact],
    config: Config,
) -> IndexArtifact:
    """Build an index over core context and observed dynamic dimensions."""

    raise NotImplementedError


def retrieve_structured(
    question: Question,
    query_context: QueryContext,
    index: IndexArtifact,
    facts_by_id: Mapping[str, FinancialFact],
    config: Config,
) -> list[FactCandidate]:
    """Return singular fact candidates without assuming a fixed scope schema."""

    raise NotImplementedError


def rank_structured_lookup(
    question: Question,
    query_context: QueryContext,
    candidates: Sequence[FactCandidate],
    config: Config,
) -> list[RankedFact]:
    """Rank B2 facts by configured structured and metric compatibility rules."""

    raise NotImplementedError

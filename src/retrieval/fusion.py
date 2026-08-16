"""Multi-route candidate fusion and exact-fact projection interfaces."""

from collections.abc import Mapping, Sequence

from src.config import Config
from src.types import (
    FactCandidate,
    FinancialFact,
    RetrievalCandidate,
    RetrievalRoute,
)


def fuse_route_candidates(
    route_candidates: Mapping[RetrievalRoute, Sequence[RetrievalCandidate]],
    config: Config,
) -> list[RetrievalCandidate]:
    """Fuse B0 evidence candidates without projecting their chunks to facts."""

    raise NotImplementedError


def fuse_fact_candidates(
    route_candidates: Mapping[RetrievalRoute, Sequence[FactCandidate]],
    config: Config,
) -> list[FactCandidate]:
    """Union and deduplicate singular fact candidates by stable source identity."""

    raise NotImplementedError


def project_fact_candidates(
    candidates: Sequence[RetrievalCandidate],
    facts_by_id: Mapping[str, FinancialFact],
    config: Config,
) -> list[FactCandidate]:
    """Project only one-fact FINANCIAL_FACT evidence; never project B0 chunk evidence."""

    raise NotImplementedError

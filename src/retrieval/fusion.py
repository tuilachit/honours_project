"""Reciprocal-rank fusion and exact-fact projection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from src.config import Config
from src.types import FactCandidate, FinancialFact, RetrievalCandidate, RetrievalRoute


def _rrf_settings(config: Config) -> tuple[int, int]:
    retrieval = config.get("retrieval")
    if not isinstance(retrieval, Mapping):
        raise ValueError("retrieval config must be a mapping")
    union = retrieval.get("candidate_union")
    if not isinstance(union, Mapping):
        raise ValueError("retrieval.candidate_union must be a mapping")
    if union.get("strategy") != "reciprocal_rank_fusion":
        raise ValueError("Only reciprocal_rank_fusion is implemented")
    return int(cast(str | int, union["rrf_k"])), int(cast(str | int, union["top_k"]))


def fuse_route_candidates(
    route_candidates: Mapping[RetrievalRoute, Sequence[RetrievalCandidate]],
    config: Config,
) -> list[RetrievalCandidate]:
    """Fuse route rankings by stable evidence identity."""

    rrf_k, top_k = _rrf_settings(config)
    grouped: dict[str, list[RetrievalCandidate]] = {}
    for candidates in route_candidates.values():
        for candidate in candidates:
            grouped.setdefault(candidate.evidence.evidence_id, []).append(candidate)
    fused: list[tuple[float, RetrievalCandidate]] = []
    for evidence_id, contributions in grouped.items():
        scores: dict[RetrievalRoute, float] = {}
        ranks: dict[RetrievalRoute, int] = {}
        for contribution in contributions:
            scores.update(contribution.route_scores)
            ranks.update(contribution.route_ranks)
        fusion_score = sum(1.0 / (rrf_k + rank) for rank in ranks.values())
        fused.append(
            (
                fusion_score,
                RetrievalCandidate(
                    question_id=contributions[0].question_id,
                    evidence=contributions[0].evidence,
                    route_scores=scores,
                    route_ranks=ranks,
                    metadata={"fusion_score": fusion_score, "evidence_id": evidence_id},
                ),
            )
        )
    return [
        candidate
        for _, candidate in sorted(
            fused, key=lambda item: (-item[0], item[1].evidence.evidence_id)
        )[:top_k]
    ]


def project_fact_candidates(
    candidates: Sequence[RetrievalCandidate],
    facts_by_id: Mapping[str, FinancialFact],
    config: Config,
) -> list[FactCandidate]:
    """Project only singular evidence units to exact facts."""

    del config
    projected: list[FactCandidate] = []
    for candidate in candidates:
        if len(candidate.evidence.fact_ids) != 1:
            raise ValueError("Exact-fact projection requires exactly one fact ID")
        fact_id = candidate.evidence.fact_ids[0]
        if fact_id not in facts_by_id:
            raise KeyError(f"Unknown fact ID in retrieval candidate: {fact_id}")
        projected.append(
            FactCandidate(
                question_id=candidate.question_id,
                fact=facts_by_id[fact_id],
                evidence_id=candidate.evidence.evidence_id,
                route_scores=candidate.route_scores,
                route_ranks=candidate.route_ranks,
                metadata=candidate.metadata,
            )
        )
    return projected


def fuse_fact_candidates(
    route_candidates: Mapping[RetrievalRoute, Sequence[FactCandidate]],
    config: Config,
) -> list[FactCandidate]:
    """Fuse already projected facts by stable source-cell identity."""

    rrf_k, top_k = _rrf_settings(config)
    grouped: dict[str, list[FactCandidate]] = {}
    for candidates in route_candidates.values():
        for candidate in candidates:
            grouped.setdefault(candidate.fact.source_address.cell_id, []).append(candidate)
    fused: list[tuple[float, FactCandidate]] = []
    for cell_id, contributions in grouped.items():
        scores: dict[RetrievalRoute, float] = {}
        ranks: dict[RetrievalRoute, int] = {}
        for contribution in contributions:
            scores.update(contribution.route_scores)
            ranks.update(contribution.route_ranks)
        fusion_score = sum(1.0 / (rrf_k + rank) for rank in ranks.values())
        fused.append(
            (
                fusion_score,
                FactCandidate(
                    question_id=contributions[0].question_id,
                    fact=contributions[0].fact,
                    evidence_id=contributions[0].evidence_id,
                    route_scores=scores,
                    route_ranks=ranks,
                    metadata={"fusion_score": fusion_score, "cell_id": cell_id},
                ),
            )
        )
    return [
        candidate
        for _, candidate in sorted(
            fused, key=lambda item: (-item[0], item[1].fact.source_address.cell_id)
        )[:top_k]
    ]

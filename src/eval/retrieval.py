"""Exact-cell retrieval and flattened-evidence coverage metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from src.config import Config
from src.types import GoldCellLabel, RankedFact, RetrievalCandidate


def _cutoffs(config: Config, key: str) -> tuple[int, ...]:
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("evaluation config must be a mapping")
    values = evaluation.get(key)
    if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
        raise ValueError(f"evaluation.{key} must be an integer list")
    return tuple(cast(list[int], values))


def compute_exact_cell_metrics(
    ranked_facts: Sequence[RankedFact],
    gold_label: GoldCellLabel,
    config: Config,
) -> Mapping[str, float]:
    """Score membership after stable first-occurrence cell deduplication."""

    seen: set[str] = set()
    cell_ids: list[str] = []
    for ranked in ranked_facts:
        cell_id = ranked.fact.source_address.cell_id
        if cell_id not in seen:
            seen.add(cell_id)
            cell_ids.append(cell_id)
    valid = set(gold_label.valid_cell_ids)
    first_rank = next((rank for rank, cell_id in enumerate(cell_ids, 1) if cell_id in valid), None)
    metrics: dict[str, float] = {
        "hit_at_1": float(bool(cell_ids) and cell_ids[0] in valid),
        "mean_reciprocal_rank": 0.0 if first_rank is None else 1.0 / first_rank,
    }
    for cutoff in _cutoffs(config, "exact_cell_cutoffs"):
        metrics[f"recall_at_{cutoff}"] = float(
            any(cell_id in valid for cell_id in cell_ids[:cutoff])
        )
    return metrics


def compute_chunk_evidence_coverage(
    candidates: Sequence[RetrievalCandidate],
    gold_label: GoldCellLabel,
    config: Config,
) -> Mapping[str, float]:
    """Measure whether any candidate chunk contains a fact from a valid cell."""

    valid = set(gold_label.valid_cell_ids)

    def contains_gold(candidate: RetrievalCandidate) -> bool:
        cell_ids = candidate.evidence.metadata.get("cell_ids", ())
        return isinstance(cell_ids, list | tuple) and bool(valid.intersection(cell_ids))

    return {
        f"evidence_coverage_at_{cutoff}": float(any(map(contains_gold, candidates[:cutoff])))
        for cutoff in _cutoffs(config, "chunk_cutoffs")
    }


def compute_stage_miss_metrics(
    candidates: Sequence[RetrievalCandidate],
    ranked_facts: Sequence[RankedFact],
    gold_label: GoldCellLabel,
    config: Config,
) -> Mapping[str, float]:
    """Separate first-stage absence from loss in the final reranked prefix."""

    del config
    valid = set(gold_label.valid_cell_ids)
    candidate_cells = {
        cell_id
        for candidate in candidates
        if isinstance((cell_id := candidate.evidence.metadata.get("cell_id")), str)
    }
    ranked_cells = {ranked.fact.source_address.cell_id for ranked in ranked_facts}
    candidate_hit = bool(valid.intersection(candidate_cells))
    final_hit = bool(valid.intersection(ranked_cells))
    return {
        "candidate_generation_miss": float(not candidate_hit),
        "reranking_miss": float(candidate_hit and not final_hit),
        "final_miss": float(not final_hit),
    }

"""Exact-cell retrieval and flattened-evidence coverage interfaces."""

from collections.abc import Mapping, Sequence

from src.config import Config
from src.types import GoldCellLabel, RankedFact, RetrievalCandidate


def compute_exact_cell_metrics(
    ranked_facts: Sequence[RankedFact],
    gold_label: GoldCellLabel,
    config: Config,
) -> Mapping[str, float]:
    """Score cell-ID membership only, after deduplicating the final ranking by cell ID."""

    raise NotImplementedError


def compute_chunk_evidence_coverage(
    candidates: Sequence[RetrievalCandidate],
    gold_label: GoldCellLabel,
    config: Config,
) -> Mapping[str, float]:
    """Measure whether flattened chunk candidates contain any valid gold cell."""

    raise NotImplementedError


def compute_stage_miss_metrics(
    candidates: Sequence[RetrievalCandidate],
    ranked_facts: Sequence[RankedFact],
    gold_label: GoldCellLabel,
    config: Config,
) -> Mapping[str, float]:
    """Separate candidate-generation misses from reranking misses."""

    raise NotImplementedError

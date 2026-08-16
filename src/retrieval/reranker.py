"""Generic and schema-flexible context-aware reranking interfaces."""

from collections.abc import Iterable, Sequence

from src.config import Config
from src.types import (
    FactCandidate,
    ModelArtifact,
    QueryContext,
    Question,
    RankedFact,
    TrainingPair,
)


def rerank_generic(
    question: Question,
    candidates: Sequence[FactCandidate],
    config: Config,
) -> list[RankedFact]:
    """Rerank exact-fact candidates with the frozen generic cross-encoder."""

    raise NotImplementedError


def rerank_context_aware(
    question: Question,
    query_context: QueryContext,
    candidates: Sequence[FactCandidate],
    model_artifact: ModelArtifact,
    config: Config,
) -> list[RankedFact]:
    """Rerank facts using core context, dynamic dimensions, and raw header paths."""

    raise NotImplementedError


def train_context_reranker(
    training_pairs: Iterable[TrainingPair],
    seed: int,
    config: Config,
) -> ModelArtifact:
    """Train one configured seed and return its fully identified model artifact."""

    raise NotImplementedError

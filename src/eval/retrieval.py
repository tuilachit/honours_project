"""Retrieval metric interfaces."""

from collections.abc import Mapping, Sequence

from src.config import Config
from src.types import Chunk, Question


def compute_retrieval_metrics(
    question: Question,
    retrieved: Sequence[Chunk],
    config: Config,
) -> Mapping[str, float]:
    """Compute configured retrieval metrics for one question."""

    raise NotImplementedError


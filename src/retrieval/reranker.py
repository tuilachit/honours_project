"""Cross-encoder reranking interface."""

from collections.abc import Sequence

from src.config import Config
from src.types import Chunk, Question


def rerank(question: Question, candidates: Sequence[Chunk], config: Config) -> list[Chunk]:
    """Rerank retrieval candidates and return the configured context window."""

    raise NotImplementedError


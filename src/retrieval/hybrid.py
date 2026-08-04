"""Hybrid retrieval interface."""

from collections.abc import Sequence

from src.config import Config
from src.types import Chunk, Question


def retrieve_hybrid(question: Question, corpus: Sequence[Chunk], config: Config) -> list[Chunk]:
    """Fuse dense and sparse candidates according to configuration."""

    raise NotImplementedError


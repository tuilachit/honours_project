"""Sparse BM25 retrieval interface."""

from collections.abc import Sequence

from src.config import Config
from src.types import Chunk, Question


def retrieve_sparse(question: Question, corpus: Sequence[Chunk], config: Config) -> list[Chunk]:
    """Return sparse-retrieval candidates in descending score order."""

    raise NotImplementedError


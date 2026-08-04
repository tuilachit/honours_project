"""Dense retrieval interface."""

from collections.abc import Sequence

from src.config import Config
from src.types import Chunk, Question


def retrieve_dense(question: Question, corpus: Sequence[Chunk], config: Config) -> list[Chunk]:
    """Return dense-retrieval candidates in descending score order."""

    raise NotImplementedError


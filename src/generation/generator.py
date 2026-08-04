"""LLM answer generation interface."""

from collections.abc import Sequence

from src.config import Config
from src.types import Answer, Chunk, Question


def generate_answer(question: Question, context: Sequence[Chunk], config: Config) -> Answer:
    """Generate an answer using the fixed prompt in the resolved condition config."""

    raise NotImplementedError

